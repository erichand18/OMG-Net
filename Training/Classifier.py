import toml
import sys
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import lightning as L
import torch
torch.set_float32_matmul_precision("medium")
from Utils import ColourAugment
import albumentations as A
import torchvision.transforms as T
from Models.SAM_Classifier import Classifier
from Dataloader.SAM import DataModule
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
from datetime import datetime
from torchvision.transforms import v2
import time

def load_config(config_file):
    return toml.load(config_file)

def get_logger(config, timestamp):
    return TensorBoardLogger(os.path.join(config['CHECKPOINT']['logger_folder'],
                                          config['CHECKPOINT']['model_name'],
                                          config['BASEMODEL']['Backbone'],
                                          ),
                             name="Mask_Input_" + str(config['BASEMODEL']['Mask_Input']), version=timestamp)
def get_callbacks(config):
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['CHECKPOINT']['logger_folder'],
        filename="{epoch:02d}-{val_f1_score:.4f}_SAM_Classifier",
        monitor="val_f1_score",
        mode="max",
        save_top_k=config['CHECKPOINT']['save_top_k'],
        save_last=config['CHECKPOINT']['save_last'],
        save_weights_only=True,
    )
    return [lr_monitor, checkpoint_callback]

def get_transforms(config):
    input_h, input_w = config['DATA']['Input_Size']

    augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=90, p=0.5),  # cleaner rotation vs. Affine+RandomRotate90
        A.RandomBrightnessContrast(0.25, 0.25, p=0.6),
        A.ColorJitter(0.15, 0.15, 0.15, 0.05, p=0.4),
        A.HueSaturationValue(10, 15, 10, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.05),  # subtle blur only
    ])
    # ✅ Validation (no randomness)
    val_augmentation = A.Compose([
        A.Resize(height=input_h, width=input_w),
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

    filenames = list(df_train_val['image_id'].unique())
    train_idx, val_idx = train_test_split(filenames, test_size=config['DATA']['val_size'])

    df_train = df[df['image_id'].isin(train_idx)]
    df_train.reset_index(drop=True, inplace=True)

    df_val = df[df['image_id'].isin(val_idx)]
    df_val.reset_index(drop=True, inplace=True)

    print('Training Size: {}/{}({}) Positive Rate: {}'.format(len(df_train), len(df), len(df_train) / len(df),
                                                              list(df_train['class'].value_counts(normalize=True))[0]))
    print('Validation Size: {}/{}({}) Positive Rate: {}'.format(len(df_val), len(df), len(df_val) / len(df),
                                                                list(df_val['class'].value_counts(normalize=True))[0]))
    print('Testing Size: {}/{}({}) Positive Rate: {}'.format(len(df_test), len(df), len(df_test) / len(df),
                                                             list(df_test['class'].value_counts(normalize=True))[0]))

    return df_train, df_val, df_test


def main(config_file):
    start = time.time()
    config = toml.load(config_file)
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")

    augmentation, val_augmentation, train_normalization, val_normalization = get_transforms(config)
    masks_dataset_train, masks_dataset_val, masks_dataset_test = get_datasets(config)
    print("-" * 50 + "Time Elapsed: {}".format(time.time() - start) + "-" * 50)

    data = DataModule(df_train = masks_dataset_train,
                      df_val = masks_dataset_val,
                      df_test = masks_dataset_test,
                      config = config,
                      train_normalization=train_normalization,
                      val_normalization=val_normalization,
                      augmentation=augmentation,
                      val_augmentation=val_augmentation,
                      inference=False,
                      )

    logger = get_logger(config, timestamp)
    print(logger.log_dir)
    callbacks = get_callbacks(config)
    L.seed_everything(config['BASEMODEL']['Random_Seed'], workers=True)

    model = Classifier(config)

    trainer = L.Trainer(devices=config['BASEMODEL']['GPU_ID'],
                        accelerator="gpu",
                        benchmark=False,
                        max_epochs=config['BASEMODEL']['Max_Epochs'],
                        callbacks=callbacks,
                        logger=logger,
                        enable_progress_bar=True,
                        precision=config['BASEMODEL']['Precision'],
                        )

    trainer.fit(model,data)
    trainer.test(ckpt_path='best', dataloaders=data.test_dataloader())

if __name__ == "__main__":
    main(sys.argv[1])