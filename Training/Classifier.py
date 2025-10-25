import toml
import sys
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import lightning as L
import torch
torch.set_float32_matmul_precision("medium")
from Utils import ColourAugment
import albumentations as A
import torchvision.transforms as T
from Models.SAM_Classifier import Classifier
from Dataloader.SAM import DataModule
import pandas as pd
import numpy as np
from datetime import datetime
from torchvision.transforms import v2
from sklearn.model_selection import StratifiedGroupKFold
import os, glob, re, time
from sklearn.metrics import precision_recall_curve, f1_score as sk_f1
# from sklearn.model_selection import train_test_split
from torch.nn.functional import softmax

def pr_f1_opt(probs, targets):
    P, R, Th = precision_recall_curve(targets, probs)
    F1 = 2 * P * R / (P + R + 1e-12)
    i = int(F1.argmax())
    thr = float(Th[max(i, 0)] if i < len(Th) else 0.5)
    return float(F1[i]), thr

def bootstrap_f1(probs, targets, B=200, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    n = len(targets)
    f1s = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        p = probs[idx]; y = targets[idx]
        f1_b, _ = pr_f1_opt(p, y)  # re-fit threshold per resample
        f1s.append(f1_b)
    f1s = np.array(f1s)
    mean = float(f1s.mean())
    se   = float(f1s.std(ddof=1) / np.sqrt(len(f1s)))
    lb95 = float(np.quantile(f1s, 0.025))  # or mean - 1.96*SE
    return mean, se, lb95

def eval_ckpt_on_val(ckpt_path, model_cls, config, val_loader, device="cuda"):
    model = model_cls.load_from_checkpoint(ckpt_path, config=config).to(device)
    model.eval()
    probs_list, t_list = [], []
    with torch.no_grad():
        for data, target in val_loader:
            for k in data: data[k] = data[k].to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(data)
            probs = softmax(logits, dim=1)[:, 1]
            probs_list.append(probs.detach().cpu().numpy())
            t_list.append(target.detach().cpu().numpy())
    probs = np.concatenate(probs_list)
    targets = np.concatenate(t_list)
    n = len(targets)
    pos_rate = float(targets.mean())
    f1_05 = sk_f1(targets, (probs >= 0.5).astype(np.int32))
    f1_opt, thr_opt = pr_f1_opt(probs, targets)  # your PR-sweep helper
    print(f"[val sanity] {os.path.basename(ckpt_path)} | N={n}, pos={pos_rate:.3f}, "
        f"F1@0.5={f1_05:.3f}, F1@opt={f1_opt:.3f}, thr={thr_opt:.3f}")
    if f1_opt > 0.95 or n < 1000 or pos_rate in (0.0, 1.0):
        raise RuntimeError("Implausible val metrics; skipping ckpt")
    f1_full, thr_full = pr_f1_opt(probs, targets)
    mean, se, lb95 = bootstrap_f1(probs, targets, B=200)
    return dict(path=ckpt_path, f1_full=f1_full, thr_full=thr_full,
                mean=mean, se=se, lb95=lb95)

def load_config(config_file):
    return toml.load(config_file)

def get_logger(config, timestamp):
    return TensorBoardLogger(os.path.join(config['CHECKPOINT']['logger_folder'],
                                          config['CHECKPOINT']['model_name'],
                                          config['BASEMODEL']['Backbone'],
                                          ),
                             name="Mask_Input_" + str(config['BASEMODEL']['Mask_Input']), version=timestamp)
def get_callbacks(config, ckpt_dir):
    lr_monitor = LearningRateMonitor(logging_interval="step")
    f1_checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:02d}-{val_f1_score:.4f}_SAM_Classifier",
        monitor="val_f1_score",
        mode="max",
        save_top_k=5,          # keep a few
        save_last=True,
        save_weights_only=True,
    )
    loss_checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:02d}-{val_loss:.4f}_SAM_Classifier",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_weights_only=True,
    )
    early_stop_f1 = EarlyStopping(
        monitor="val_f1_score",
        mode="max",
        patience=10,
        min_delta=1e-3,
        verbose=True,
        stopping_threshold=config['BASEMODEL'].get("F1_Stop_Threshold", 0.9),
        check_on_train_epoch_end=False,
    )
    early_stop_loss = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=15,
        min_delta=1e-4,
        verbose=True,
        check_on_train_epoch_end=False,
    )
    return [lr_monitor, f1_checkpoint, loss_checkpoint, early_stop_f1, early_stop_loss]

def get_transforms(config):
    input_h, input_w = config['DATA']['Input_Size']

    augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=90, p=0.5),
        A.RandomBrightnessContrast(0.25, 0.25, p=0.6),
        A.ColorJitter(0.15, 0.15, 0.15, 0.05, p=0.4),
        A.HueSaturationValue(10, 15, 10, p=0.4),
    ])
    # ✅ Validation (no randomness or resizing)
    val_augmentation = A.Compose([
        # Intentionally empty. The DataGenerator handles cropping.
        # Resizing here would be incorrect.
    ])

    # ✅ Normalization (Torchvision)
    train_normalization = T.Compose([
        T.ToTensor(),
        ColourAugment.ColourAugment(
            sigma=config['AUGMENTATION']['Colour_Sigma'],
            mode=config['AUGMENTATION']['Colour_Mode']
        ),
        v2.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    val_normalization = T.Compose([
        T.ToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    return augmentation, val_augmentation, train_normalization, val_normalization

def get_datasets(config):
    df = pd.read_csv(config['DATA']['Dataframe'])
    df["image_id"] = df["image_id"].astype(str).str.zfill(3)
    df = df[(df['image_quality'] == 1) & (df['class'] != 999)]
    df = df[df['species'].isin(config['DATA']['species'])]
    df = df[df['source'].isin(config['DATA']['datasets'])]
    df.reset_index(drop=True, inplace=True)
    print(df)

    df_test = df[df['image_id'].isin(config['DATA']['filenames_test'])].reset_index(drop=True)
    df_train_val = df[~df['image_id'].isin(config['DATA']['filenames_test'])].reset_index(drop=True)

    gkf = StratifiedGroupKFold(
        n_splits=int(1 / config['DATA']['val_size']),
        shuffle=True,
        random_state=config['BASEMODEL']['Random_Seed']
    )
    # Ensure groups are handled correctly to prevent leakage
    split_idx = next(gkf.split(df_train_val, df_train_val['class'], groups=df_train_val['image_id']))
    train_idx, val_idx = split_idx

    # Create dataframes from the correct indices and source (df_train_val)
    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val   = df_train_val.iloc[val_idx].reset_index(drop=True)

    # --- Sanity checks ---
    def pos_rate(df_):
        return df_['class'].value_counts(normalize=True).get(1, 0.0)

    print(f"Training Size:   {len(df_train)}/{len(df)} ({len(df_train)/len(df):.2%})  "
          f"Positive Rate: {pos_rate(df_train):.3f}")
    print(f"Validation Size: {len(df_val)}/{len(df)} ({len(df_val)/len(df):.2%})  "
          f"Positive Rate: {pos_rate(df_val):.3f}")
    print(f"Testing Size:    {len(df_test)}/{len(df)} ({len(df_test)/len(df):.2%})  "
          f"Positive Rate: {pos_rate(df_test):.3f}")

    return df_train, df_val, df_test


def main(config_file):
    start = time.time()
    config = toml.load(config_file)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    augmentation, val_augmentation, train_normalization, val_normalization = get_transforms(config)
    df_train, df_val, df_test = get_datasets(config)
    print("-" * 50 + f"Time Elapsed: {time.time() - start}" + "-" * 50)

    data = DataModule(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        config=config,
        train_normalization=train_normalization,
        val_normalization=val_normalization,
        augmentation=augmentation,
        val_augmentation=val_augmentation,
        inference=False,
    )

    logger = get_logger(config, timestamp)
    ckpt_dir = os.path.join(logger.log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = get_callbacks(config, ckpt_dir)
    f1_ckpt = [cb for cb in callbacks if isinstance(cb, ModelCheckpoint) and cb.monitor == "val_f1_score"][0]
    L.seed_everything(config["BASEMODEL"]["Random_Seed"], workers=True)

    model = Classifier(config)

    trainer = L.Trainer(
        devices=config["BASEMODEL"]["GPU_ID"],
        accelerator="gpu",
        benchmark=False,
        max_epochs=config["BASEMODEL"]["Max_Epochs"],
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        precision=config["BASEMODEL"]["Precision"],
    )

    trainer.fit(model, data)

    candidate_dirs = {ckpt_dir}
    patterns = [os.path.join(ckpt_dir, "*.ckpt")]
    ckpt_paths = sorted({p for pat in patterns for p in glob.glob(pat)})
    if not ckpt_paths and getattr(f1_ckpt, "best_model_path", ""):
        ckpt_paths = [f1_ckpt.best_model_path]

    results = []
    for p in ckpt_paths:
        try:
            r = eval_ckpt_on_val(p, Classifier, config, data.val_dataloader())
            m = re.search(r"epoch=(\d+)", p)
            r["epoch"] = int(m.group(1)) if m else 10**9
            results.append(r)
        except Exception as e:
            print(f"[skip] {p}: {e}")

    if results:
        chosen = max(results, key=lambda r: r["lb95"])
        chosen_path = chosen["path"]
        chosen_thr  = float(chosen["thr_full"])
        print(f"\nSelected checkpoint (best lb95): {os.path.basename(chosen_path)} "
            f"| val_mean={chosen['mean']:.4f}, SE={chosen['se']:.4f}, lb95={chosen['lb95']:.4f}, thr={chosen_thr:.3f}")
    else:
        chosen_path = getattr(f1_ckpt, "best_model_path", "")
        if not chosen_path:
            raise RuntimeError("No checkpoints found to evaluate.")
        chosen_thr = 0.5
        print(f"\nSelected fallback checkpoint: {os.path.basename(chosen_path)} | thr={chosen_thr:.3f}")

    model_sel = Classifier.load_from_checkpoint(chosen_path, config=config)
    setattr(model_sel, "best_val_threshold", chosen_thr)
    trainer.test(model=model_sel, dataloaders=data.test_dataloader())

if __name__ == "__main__":
    main(sys.argv[1])