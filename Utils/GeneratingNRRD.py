import json
import os
# import gc
import glob
# from tqdm import tqdm
import pandas as pd
import numpy as np
# import cv2
import openslide
# import sqlite3
# import matplotlib.pyplot as plt
import nrrd
import torch
from segment_anything import sam_model_registry, SamPredictor

def get_img_msk(
        filename,
        center_x, 
        center_y, 
        label, 
        nrrd_file, 
        image_folder, 
        nrrd_folder, 
        predictor, 
        dim=[256, 256], 
        vis_level=0,
        box_dim=[32, 32],
        custom_field_map=None,
        skip_oob=True,
    ):
    """Extract a patch and SAM mask safely from a WSI (level-aware, bounds-checked)."""

    # --- basic sanitization ---
    if center_x is None or center_y is None:
        print(f"⚠️  {filename}: center is None → skipping")
        return
    if np.isnan(center_x) or np.isnan(center_y):
        print(f"⚠️  {filename}: center is NaN → skipping")
        return

    # Ensure ints
    center_x = int(round(float(center_x)))
    center_y = int(round(float(center_y)))
    patch_w, patch_h = int(dim[0]), int(dim[1])

    slide_path = os.path.join(image_folder, filename)
    try:
        slide = openslide.open_slide(slide_path)
    except Exception as e:
        print(f"❌ Could not open slide {slide_path}: {e}")
        return

    slide_w, slide_h = slide.dimensions
    # Scale factor from level 0 → requested level
    try:
        scale = float(slide.level_downsamples[vis_level])
    except Exception:
        scale = 1.0

    # Size in *level-0* pixels for this patch
    req_w_l0 = int(round(patch_w * scale))
    req_h_l0 = int(round(patch_h * scale))

    # Top-left in level-0
    tl_x_l0 = int(center_x - req_w_l0 // 2)
    tl_y_l0 = int(center_y - req_h_l0 // 2)

    # Bounds check (level-0 space)
    out_of_bounds = (
        tl_x_l0 < 0 or tl_y_l0 < 0 or
        tl_x_l0 + req_w_l0 > slide_w or
        tl_y_l0 + req_h_l0 > slide_h
    )
    if out_of_bounds:
        if skip_oob:
            print(f"⚠️  Skipping {filename}: L0 region ({tl_x_l0},{tl_y_l0}) "
                  f"size ({req_w_l0},{req_h_l0}) exceeds slide ({slide_w},{slide_h}) at level {vis_level}")
            slide.close()
            return
        # Clamp to fit
        tl_x_l0 = max(0, min(tl_x_l0, slide_w - req_w_l0))
        tl_y_l0 = max(0, min(tl_y_l0, slide_h - req_h_l0))

    # For predictor, we need center in patch coordinates at requested level
    rel_cx = int(round((center_x - tl_x_l0) / scale))
    rel_cy = int(round((center_y - tl_y_l0) / scale))

    # Safety: keep SAM bbox fully inside patch
    half_box_w = int(box_dim[0] // 2)
    half_box_h = int(box_dim[1] // 2)
    rel_cx = max(half_box_w, min(rel_cx, patch_w - half_box_w - 1))
    rel_cy = max(half_box_h, min(rel_cy, patch_h - half_box_h - 1))

    # Try the read; if it still fails, skip gracefully
    try:
        region = slide.read_region((tl_x_l0, tl_y_l0), vis_level, (patch_w, patch_h))
        img = np.array(region, copy=False)[:, :, :3]
    except Exception as e:
        print(f"❌ read_region failed for {filename} at L0 ({tl_x_l0},{tl_y_l0}) "
              f"size ({patch_w},{patch_h}) level {vis_level}: {e}")
        slide.close()
        return
    finally:
        slide.close()

    # Build bbox in patch (requested level) pixels
    bbox = [
        rel_cx - half_box_w,
        rel_cy - half_box_h,
        rel_cx + half_box_w,
        rel_cy + half_box_h,
    ]

    # Run SAM
    try:
        predictor.set_image(img)
        masks, _, _ = predictor.predict(box=np.array([bbox]), multimask_output=False)
        mask = masks[0].astype("float32")
    except Exception as e:
        print(f"❌ SAM failed on {filename} ({bbox}): {e}")
        return

    header = {
        'filename': filename,
        'top_left': (tl_x_l0, tl_y_l0),         # stored in level-0 coords
        'center': (rel_cx, rel_cy),             # patch coords at vis_level
        'dim': (patch_w, patch_h),
        'vis_level': int(vis_level),
        'annotation_label': 1 if label == 'mitotic figure' else 0,
        'mask': mask,
    }

    # Ensure output dir
    os.makedirs(nrrd_folder, exist_ok=True)

    try:
        nrrd.write(os.path.join(nrrd_folder, nrrd_file), img.astype(np.uint8),
                   header, custom_field_map=custom_field_map)
        print(f"✅ Saved {nrrd_file} from {filename} at L0 ({tl_x_l0},{tl_y_l0}), level {vis_level}")
    except Exception as e:
        print(f"❌ NRRD write failed for {nrrd_file}: {e}")

def save_nrrd_from_df(df, image_folder, nrrd_folder, predictor, dim=[256, 256], vis_level=0, box_dim=[32, 32]):
    for i, row in df.reset_index(drop=True).iterrows():
        try:
            get_img_msk(
                row['filename'], row['coordinateX'], row['coordinateY'],
                row['annotation_label'], row['nrrd_file'],
                image_folder, nrrd_folder, predictor,
                dim=dim, vis_level=vis_level, box_dim=box_dim,
                custom_field_map=custom_field_map, skip_oob=True
            )
        except Exception as e:
            print(f"❌ Row {i} ({row['filename']}) failed: {e}")

def table2df(cursor, table_name):
    cursor.execute(f"SELECT * FROM {table_name};")
    rows = cursor.fetchall()
    columns = [column[0] for column in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    return df

#Configure Paths
nrrd_folder        ="/nrrd"                #Path to save the NRRD files
final_dataset      = "/final_dataset"       #Path to save the CSV file of the final dataset
midog_folder       = "/MIDOG_PLUS/"         #Path to MIDOG_PLUS
# cmc_folder         = "/path/to/MITOS_WSI_CMC/"      #Path to MITOS_WSI_CMC
# ccmct_folder       = "/path/to/MITOS_WSI_CCMCT/"    #Path to MITOS_WSI_CCMCT
# tupac_folder       = "/path/to/TUPAC/"              #Path to TUPAC

#Load SAM mask generator
sam_checkpoint     = "weights/sam_vit_h_4b8939.pth" 
model_type         = "vit_h"
device             = "cuda"
dim                = [256, 256]
vis_level          = 0
box_dim            = [32, 32]
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=torch.device(device))
predictor = SamPredictor(sam)

#Configure NRRD files
custom_field_map   = {
    'SVS_ID': 'string',
    'top_left': 'int list',
    'center': 'int list',
    'dim': 'int list',
    'vis_level': 'int',
    'diagnosis': 'string',
    'annotation_label': 'string',
    'mask': 'double matrix'
    }

datasets = []

#Process the MIDOG dataset-----------------------------------------------------------------------------------------------
print("Processing MIDOG dataset")
annotation_file = os.path.join(midog_folder, "MIDOG++.json")
image_folder    = midog_folder
image_files     = [fn.split("\\")[-1] for fn in glob.glob(image_folder + "/*.tiff")]
slides          = pd.read_csv(os.path.join(midog_folder, "datasets_xvalidation.csv"), delimiter=";")
dataframe       = os.path.join(midog_folder, "MIDOG.csv")

print("Creating dataframe from annotation")
rows = []
with open(annotation_file) as f:
    data = json.load(f)
    categories = {1: 'mitotic figure', 2: 'hard negative'}
    for row in data["images"]:
        file_name = row["file_name"]
        image_id = row["id"]
        width = row["width"]
        height = row["height"]
        for ann_id, annotation in enumerate([anno for anno in data['annotations'] if anno["image_id"] == image_id]):
            box = annotation["bbox"]
            xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cat = categories[annotation["category_id"]]
            slide = slides.loc[slides['Slide'] == image_id, 'Tumor':]
            tumour = slide['Tumor'].array[0]
            scanner = slide['Scanner'].array[0]
            origin = slide['Origin'].array[0]
            species = slide['Species'].array[0]
            nrrd_file = 'MIDOG_{}_{}.nrrd'.format(f"{image_id:03d}", ann_id)
            if annotation["image_id"] > 273:
                rows.append(
                    [file_name, image_id, ann_id, width, height, xmin, ymin, xmax, ymax, cat, tumour, scanner, origin,
                        species, nrrd_file])
            else: 
                print("skipping image id:", annotation["image_id"])

df = pd.DataFrame(rows, columns=["filename", "image_id", "ann_id", "width", "height",
                                 "xmin", "ymin", "xmax", "ymax",
                                 "annotation_label", "tumour", "scanner", "origin", "species", "nrrd_file"])
df['coordinateX'] = (df['xmin'] + df['xmax']) / 2
df['coordinateY'] = (df['ymin'] + df['ymax']) / 2
df.to_csv(dataframe, index=False)

save_nrrd_from_df(df, image_folder, nrrd_folder, predictor, dim, vis_level, box_dim)
datasets.append(df)


# #Process the MITOS_WSI_CMC/MITOS_WSI_CCMCT dataset-----------------------------------------------------------------------------------------------
# print("Processing MITOS_WSI_CMC/MITOS_WSI_CCMCT dataset")
# for dataset in ['MITOS_WSI_CMC', 'MITOS_WSI_CCMCT']:
#     if dataset == "MITOS_WSI_CCMCT":
#         annotation_file = os.path.join(ccmct_folder, "databases", "MITOS_WSI_CCMCT_ODAEL.sqlite")
#         image_folder    = os.path.join(ccmct_folder, "WSI")
#         dataframe       = os.path.join(ccmct_folder, "MITOS_WSI_CCMCT.csv")
#         mitotic_label   = 'mitotic figure'

#     elif dataset == "MITOS_WSI_CMC":
#         annotation_file = os.path.join(cmc_folder, "databases", "MITOS_WSI_CMC_CODAEL_TR_ROI.sqlite")
#         image_folder    = os.path.join(cmc_folder, "WSI")
#         dataframe       = os.path.join(cmc_folder, "MITOS_WSI_CMC.csv")
#         mitotic_label   = 'Mitotic figure'
    
#     print("Creating dataframe from annotation")
#     con = sqlite3.connect(annotation_file)
#     cur = con.cursor()
#     cur.execute("SELECT name FROM sqlite_master WHERE type='table';")

#     df_Annotations = table2df(cur, 'Annotations')
#     df_sqlite_sequence = table2df(cur, 'sqlite_sequence')
#     df_Annotations_coordinates = table2df(cur, 'Annotations_coordinates')
#     df_Annotations_label = table2df(cur, 'Annotations_label')
#     df_Classes = table2df(cur, 'Classes')
#     df_Log = table2df(cur, 'Log')
#     df_Persons = table2df(cur, 'Persons')
#     df_Slides = table2df(cur, 'Slides')
#     con.close()

#     df_Annotations = df_Annotations[df_Annotations['agreedClass'].isin([1, 2])]
#     df_Annotations_coordinates = df_Annotations_coordinates[df_Annotations_coordinates['orderIdx'] == 1]
#     df_Annotations_coordinates.drop(columns=['slide'], inplace=True)
#     df_Annotations = df_Annotations.rename(columns={'uid': 'annoId'})

#     df_Slides = df_Slides.rename(columns={'uid': 'slide'})
#     df_Classes = df_Classes.rename(columns={'uid': 'agreedClass', 'name': 'annotation_label'})

#     df = df_Annotations.merge(df_Slides, on='slide', how='inner')
#     df = df.merge(df_Annotations_coordinates, on='annoId', how='inner')
#     df = df.merge(df_Classes, on='agreedClass', how='inner')
#     df = df.replace(mitotic_label, 'mitotic figure')
#     df.reset_index(drop=True, inplace=True)
#     df['nrrd_file'] = ['{}_{}.nrrd'.format(df['filename'][i].split(".")[0], df['annoId'][i]) for i in range(len(df))]
#     df.to_csv(dataframe, index=False)

#     save_nrrd_from_df(df, image_folder, nrrd_folder, predictor, dim, vis_level, box_dim)
#     datasets.append(df)

# #Process the TUPAC16 dataset-----------------------------------------------------------------------------------------------
# print("Processing TUPAC16 dataset")
# annotation_file = os.path.join(tupac_folder, "databases", "TUPAC_alternativeLabels_augmented_training.sqlite")
# image_folder    = os.path.join(tupac_folder, "WSI")
# dataframe       = os.path.join(tupac_folder, "TUPAC16.csv")

# print("Creating dataframe from annotation")
# con = sqlite3.connect(annotation_file)
# cur = con.cursor()
# cur.execute("SELECT name FROM sqlite_master WHERE type='table';")

# df_Annotations = table2df(cur, 'Annotations')
# df_sqlite_sequence = table2df(cur, 'sqlite_sequence')
# df_Annotations_coordinates = table2df(cur, 'Annotations_coordinates')
# df_Annotations_label = table2df(cur, 'Annotations_label')
# df_Classes = table2df(cur, 'Classes')
# df_Log = table2df(cur, 'Log')
# df_Persons = table2df(cur, 'Persons')
# df_Slides = table2df(cur, 'Slides')

# print(df_Annotations.agreedClass.value_counts())
# print(df_Annotations.uid.unique().shape)
# print(df_Annotations_label.annoId.unique().shape)
# print(df_Annotations_coordinates.annoId.unique().shape)
# con.close()

# df_Annotations_coordinates.drop(columns=['slide'], inplace=True)
# df_Annotations = df_Annotations.rename(columns={'uid': 'annoId'})
# df_Slides = df_Slides.rename(columns={'uid': 'slide'})
# df_Classes = df_Classes.rename(columns={'uid': 'agreedClass', 'name': 'annotation_label'})

# df = df_Annotations.merge(df_Slides, on='slide', how='inner')
# df = df.merge(df_Annotations_coordinates, on='annoId', how='inner')
# df = df.merge(df_Classes, on='agreedClass', how='inner')
# df.reset_index(drop=True, inplace=True)
# df.drop(columns=['guid', 'lastModified', 'deleted', 'type',
#                     'description', 'directory', 'uuid', 'exactImageID',
#                     'EXACTUSER', 'uid', 'orderIdx', 'coordinateZ', 'color'], inplace=True)
# df = df.replace('Mitose', 'mitotic figure')
# df['nrrd_file'] = ['{}_{}.nrrd'.format(df['filename'][i].split(".")[0], df['annoId'][i]) for i in range(len(df))]
# df.to_csv(dataframe, index=False)

# save_nrrd_from_df(df, image_folder, nrrd_folder, predictor, dim, vis_level, box_dim)
# datasets.append(df)

#Merge all datasets-----------------------------------------------------------------------------------------------
print("Merging datasets")
df = pd.concat(datasets)
df.to_csv(os.path.join(final_dataset, "final_dataset.csv"), index=False)
print("Done")



