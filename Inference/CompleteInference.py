import sys, os
import multiprocessing as mp
sys.path.append(os.getcwd())
from Dataloader.Dataloader import DataGenerator
from Dataloader.SAM import DataGeneratorInf
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2
import torch
import time
import lightning as L
import pandas as pd
from Models.SAM_Classifier import Classifier
from Models.SAM_Masking import MaskGenerator
import numpy as np
from Utils import TileOps

## Globals
n_gpus      = torch.cuda.device_count()
devices     = [0]
num_workers = 16
n_ensemble  = 1
PROCESSING_CHECKPOINT = None
def load_config():
    config = {
        'BASEMODEL': {
            'Image_Type': ".svs",
            'WSIReader': "cuCIM",
            'Input_Size': [64, 64],
            'Patch_Size': [256, 256],
            'Mask_Input': True,
            'Precision': '16-mixed',
            'Vis': [0],
            'Batch_Size_Preprocessing': 128,
            'Batch_Size_Masking': 1,
            'Batch_Size_Classification': 256,
            'Prob_Tumour_Tresh': 0.85
        },
        'SAM_MODEL': {
            'points_per_side': 32,
            'pred_iou_thresh': 0.8,
            'stability_score_thresh': 0.8,
            'box_nms_thresh': 0.1,
            'min_mask_region_area': 36,
            'max_mask_region_area': 3600,
            'Points_Batch_Size': 16
        },
        'OVERALL': {
            'Patch_Size': [512, 512],
            'Tile_Overlap': 0,
            'Background_Threshold': 0.8,
            'WSIReader': "cuCIM"
        },
        'CLASSIFICATION_MODEL': {
            'Input_Size': [64, 64],
            'Num_Classes': 2,
            'Threshold': 0.5
        }
    }
    return config


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def RemoveBackground(local_SVS_PATH, config):
    tile_coords_no_background  = TileOps.get_nonbackground_tiles(local_SVS_PATH, config)
    tile_dataset_preprocessing = pd.DataFrame({'coords_x': tile_coords_no_background[:, 0],
                                               'coords_y': tile_coords_no_background[:, 1]})  
    tile_dataset_preprocessing['SVS_PATH'] = local_SVS_PATH

    ## Manually sample
    tile_dataset_preprocessing = tile_dataset_preprocessing.sample(frac=1, random_state=42).reset_index(drop=True) ## Balancing
    tiles_per_gpu = len(tile_dataset_preprocessing) // trainer.world_size
    start_idx     = trainer.global_rank * tiles_per_gpu
    end_idx = start_idx + tiles_per_gpu if trainer.global_rank < trainer.world_size - 1 else len(tile_dataset_preprocessing)
    tile_dataset_preprocessing = tile_dataset_preprocessing[start_idx:end_idx]

    print(trainer.global_rank, start_idx, end_idx)    
    return tile_dataset_preprocessing   

def MaskGeneration(tile_dataset_preprocessing, SAM_CHECKPOINT, config):
    # tile_dataset_preprocessing.reset_index(drop=True, inplace=True)
    mask_transform = v2.Compose([v2.ToImage(),
                                 v2.ToDtype(torch.half, scale=False)])
    model_maskgenerator = MaskGenerator(config, SAM_CHECKPOINT)
    model_maskgenerator.eval()
    model_maskgenerator = freeze_model(model_maskgenerator)
    
    # Create dataloader
    data = DataLoader(DataGenerator(tile_dataset_preprocessing, config, transform = mask_transform),
                      batch_size=config['BASEMODEL']['Batch_Size_Masking'],
                      num_workers=num_workers,
                      pin_memory=False,
                      shuffle=False)

    predictions         = trainer.predict(model_maskgenerator, data)

    cropped_masks       = torch.cat([cropped_mask for cropped_mask, center, idx in predictions], dim=0)##[NMask, H, W]
    centers             = torch.cat([center for cropped_mask, center, idx in predictions], dim=0) ## [NMasks,2]
    indexes             = torch.cat([idx for cropped_mask, center, idx in predictions], dim=0) ## [NMask ]
    
    return cropped_masks, centers, indexes

def CellClassification(trainer, tile_dataset_preprocessing, cropped_masks, centers, indexes, CLASSIFY_CHECKPOINT, config):
    classif_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tile_dataset_preprocessing.reset_index(drop=True, inplace=True)
    # Create dataloader
    data = DataLoader(DataGeneratorInf(config, tile_dataset_preprocessing,
                                       masks=cropped_masks,
                                       centers=centers,
                                       indexes = indexes,
                                       transform=classif_transform),
                      batch_size=config['BASEMODEL']['Batch_Size_Classification'],
                      num_workers=num_workers,
                      shuffle=False,
                      pin_memory=False)
    del cropped_masks

    # Load model and freeze layers

    cell_type_predictions_all_models = []
    coords_all_models  = []
    
    for i in range(n_ensemble):
        model_classifier = Classifier.load_from_checkpoint(CLASSIFY_CHECKPOINT[i])
        model_classifier.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):  # disable AMP to avoid numeric skew
            batch = next(iter(data))        # get one batch from the dataloader
            # Your inference dataset yields a dict, e.g. {'img': tensor, 'msk': tensor, 'coords': tensor, ...}
            if isinstance(batch, dict):
                inputs = batch
            elif isinstance(batch, (list, tuple)) and isinstance(batch[0], dict):
                inputs = batch[0]
            else:
                raise TypeError(f"Unexpected batch type: {type(batch)}")

            print("batch keys:", list(inputs.keys()))
            print("batch keys:", inputs.keys())  # if it's a dict, e.g. {'img', 'msk'}
            for k, v in inputs.items():
                print(k, v.shape, v.dtype)

            with torch.no_grad():
                device = next(model_classifier.parameters()).device
                for k, v in inputs.items():
                    if torch.is_tensor(v):
                        inputs[k] = v.to(device)
                        if v.dtype == torch.float16:  # ensure same precision as model weights
                            inputs[k] = inputs[k].float()

                logits = model_classifier(inputs)
                probs = torch.softmax(logits, dim=1)
                print("sample logits:", logits[:5].cpu().numpy())
                print("sample probs:", probs[:5].cpu().numpy())

        model_classifier = freeze_model(model_classifier)

        predictions                     = trainer.predict(model_classifier, data)
        cell_type_predictions           = torch.cat([prob for prob, coords, idx in predictions], dim=0)
        coords                          = torch.cat([coords for prob, coords, idx in predictions], dim=0)
        indexes                         = torch.cat([idx for prob, coords, idx in predictions], dim=0)
        
        cell_type_predictions           = cell_type_predictions.view(-1, cell_type_predictions.shape[-1])
        coords                          = coords.view(-1, coords.shape[-1])
        indexes                         = indexes.view(-1)
        
        cell_type_predictions           = cell_type_predictions[indexes]
        coords                          = coords[indexes]
        cell_type_predictions_all_models.append(cell_type_predictions)
        coords_all_models.append(coords)
    
    cell_type_predictions = torch.mean(torch.stack(cell_type_predictions_all_models).float(), dim=0)
    coords                = torch.mean(torch.stack(coords_all_models).float(), dim=0)    
        
    return cell_type_predictions, coords

def complete_inference(config, trainer, local_SVS_PATH, local_result, PROCESSING_CHECKPOINT, SAM_CHECKPOINT, CLASSIFY_CHECKPOINT):
    try:
        print('1. Remove non-background tiles')
        time_start = time.time()
        tile_dataset_preprocessing = RemoveBackground(local_SVS_PATH, config)
        time_end = time.time()
        if trainer.is_global_zero:
            print(f'{len(tile_dataset_preprocessing)} tiles to classify after background removal [{int(time_end - time_start)} seconds]')
    except Exception as e:
        raise RuntimeError("Critical failure during background removal") from e

    if not os.path.exists(f'{local_result[:-4]}-masks.pth'):
        try:   
            print('2. Mask Generation')
            cropped_masks, centers, indexes = MaskGeneration(tile_dataset_preprocessing, SAM_CHECKPOINT, config)
            time_maskgeneration = time.time()
            if trainer.is_global_zero:  
                print(f'Mask generation completed [{int(time_maskgeneration - time_end)} seconds]')
            torch.save({'cropped_masks': cropped_masks, 'centers': centers, 'indexes': indexes}, f'{local_result[:-4]}-masks.pth')
            print(f'Masks saved at {local_result[:-4]}-masks.pth !')
        except Exception as e:
            raise RuntimeError("Critical failure during mask generation") from e
    else:
        print(f'Loading Masks from {local_result[:-4]}-masks.pth ... ')
        masks_file = torch.load(f'{local_result[:-4]}-masks.pth')
        cropped_masks, centers, indexes = masks_file['cropped_masks'], masks_file['centers'], masks_file['indexes']

    try:
        print('3. Classification')
        cell_type_predictions, coords =  CellClassification(trainer, tile_dataset_preprocessing, cropped_masks, centers, indexes, CLASSIFY_CHECKPOINT, config)
    except Exception as e:
        raise RuntimeError("Critical failure during mask classification") from e
        
    masks_dataset             = pd.DataFrame()
    coords                    = coords.numpy().astype(int)
    masks_dataset['coords_x'] = coords[:, 0]
    masks_dataset['coords_y'] = coords[:, 1]
    masks_dataset['SVS_PATH'] = local_SVS_PATH

    for n, class_label in enumerate(['pred_0', 'pred_1']):
        masks_dataset[class_label] = cell_type_predictions[:, n]

    np.savez(f"{local_result[:-4]}_{trainer.global_rank}.npz", masks=cropped_masks.numpy(), coords=coords)
    
    masks_dataset = masks_dataset[masks_dataset['pred_1']>0.5]

    masks_dataset.to_csv(f"{local_result[:-4]}_{trainer.global_rank}.csv", index=False)
    print(masks_dataset)
    print(f"{local_result[:-4]}_{trainer.global_rank}.csv Saved")
    trainer.strategy.barrier() ##Synchronize all the save

    if trainer.is_global_zero:
        dataset_dict = {}
        npz_dict = {}
        for gpu_id in range(trainer.world_size):
            npz_dict[f"{gpu_id}"] = np.load(f"{local_result[:-4]}_{gpu_id}.npz", allow_pickle=True)
            dataset = pd.read_csv(f"{local_result[:-4]}_{gpu_id}.csv")
            print(f"{local_result[:-4]}_{gpu_id}.csv")
            print(f"{local_result[:-4]}_{gpu_id}.npz")
            if not dataset.empty:
                dataset_dict[f"{gpu_id}"] = dataset  

        masks = np.concatenate([npz_dict[f"{gpu_id}"]['masks'] for gpu_id in range(trainer.world_size)])
        coords = np.concatenate([npz_dict[f"{gpu_id}"]['coords'] for gpu_id in range(trainer.world_size)]).astype(int)
        np.savez(f"{local_result[:-4]}.npz", masks=masks, coords = coords)

        if dataset_dict:
            masks_dataset = pd.concat([value for key, value in dataset_dict.items()], axis=0)
        else:
            masks_dataset = pd.DataFrame()
            
        masks_dataset.to_csv(f"{local_result[:-4]}.csv", index=False)
        print(masks_dataset)
        print(f"Number of cells: {masks.shape[0]}, Number of mitotic figures: {len(masks_dataset)}")
    
    return True

if __name__ == "__main__":

    config          = load_config()

    ## Checkpoints
    SAM_CHECKPOINT          = "/app/weights/sam_vit_h_4b8939.pth"
    CLASSIFY_CHECKPOINT = ["/checkpoints/epoch=27-val_loss=0.3153_SAM_Classifier.ckpt"]

    ckpt = torch.load(CLASSIFY_CHECKPOINT[0], map_location="cpu")
    sd = ckpt.get("state_dict", {})
    first = next(iter(sd.values()))
    print("keys in ckpt:", ckpt.keys())
    print("state_dict tensors:", len(sd))
    print("first tensor mean|std:", float(next(iter(sd.values())).float().abs().mean()), float(next(iter(sd.values())).float().std()))
    
    local_SVS_PATH  = sys.argv[1]
    local_result    = f"{os.path.basename(local_SVS_PATH)[:-4]}.csv"

    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')
    trainer = L.Trainer(
        devices=devices,
        accelerator="gpu",
        strategy="auto",
        logger=True,
        precision="32-true",
    )

    # Run model
    trainer.strategy.barrier() ## Sync everything to make sure the data is correctly downloaded
    signal = complete_inference(config,
                       trainer,
                       local_SVS_PATH,
                       local_result,
                       PROCESSING_CHECKPOINT,
                       SAM_CHECKPOINT,
                       CLASSIFY_CHECKPOINT)


