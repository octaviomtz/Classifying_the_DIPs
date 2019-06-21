# 1. The images were finally resampled in the slice direction (dicom full-preprocessing (misc1) v3.6 - interpolate only 2axes to inpaint - get the pylidc characteristics)
# 1. The coordinates are taken from the pylidc (after transforming to the coords in small cubes) because the coords obtained from the masks are dilated and there are cases where two nodules merge and they would be considered as one nodule
import os
from statistics import mode, StatisticsError
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
from tqdm import tqdm
# import pylidc as pl # module for handling the LIDC dataset

# ## Functions

def box_with_masks_search(coords_Z, coords_X, coords_Y, mask_lungs_, min_coords, dist1 = 64, dist2 = 64, dist3 = 64):
    '''Finds the cube containing the nodule and as many voxels from inside the lung as possible'''
    
    coords_Z_min, coords_Z_max, coords_X_min, coords_X_max, coords_Y_min, coords_Y_max = min_coords
#     print(coords_Z_min, coords_Z_max, coords_X_min, coords_X_max, coords_Y_min, coords_Y_max)
    box_found= False
    # find where the vol_cut get more info voxels
    max_sum = 0
    for i in range(21):
        ii = i * 3 - 30
        for j in range(21):
            jj = j * 3 - 30
            for k in range(21):
                kk = k * 3 - 30
                
                # limits of the current box
                zmin = int(coords_Z-(dist1//2)+ii)
                zmin = np.max([zmin, 0]); zmax = int(zmin + dist1)
                if zmax >= np.shape(mask_lungs_)[0]: continue
                
                xmin = int(coords_X-(dist2//2)+jj); 
                xmin = np.max([xmin, 0]); xmax = int(xmin + dist2)
                if xmax >= np.shape(mask_lungs_)[1]: continue
                
                ymin = int(coords_Y-(dist3//2)+kk); 
                ymin = np.max([ymin, 0]); ymax = int(ymin + dist3)
                if ymax >= np.shape(mask_lungs_)[2]: continue
                
#                 print(zmin, zmax, xmin, xmax, ymin, ymax)
                
                #if the current box contains the masks
                if zmin <= coords_Z_min and zmax >= coords_Z_max and xmin <= coords_X_min and xmax >= coords_X_max and ymin <= coords_Y_min and ymax >= coords_Y_max:
                    #print(f'if 1, {zmin, zmax, xmin, xmax, ymin, ymax}')
                    vol_cut=mask_lungs_[zmin:zmax,xmin:xmax,ymin:ymax]
                    # the box contains as many info voxels as possible
                    this_sum = np.sum(vol_cut)
                    if this_sum > max_sum:
                        #print(f'if 2, {zmin, zmax, xmin, xmax, ymin, ymax}')
                        max_sum = this_sum
                        box_found = True
                        z_min_found = zmin
                        z_max_found = zmax
                        x_min_found = xmin
                        x_max_found = xmax
                        y_min_found = ymin                        
                        y_max_found = ymax 
    if box_found == False:
        z_min_found, z_max_found, x_min_found, x_max_found, y_min_found, y_max_found = -1, 1, 1, 1, 1, 1
    return z_min_found, z_max_found, x_min_found, x_max_found, y_min_found, y_max_found        

def transform_malignancy(i):
    '''Causey et al. Highly accurate model for prediction of lung nodule malignancy with CT scans
     We tested two designs: S1 versus S45, and S12 versus S45
     Here we discard all 3's and as long as there is another number the nodule is classified as malignant or beningn
     '''
    if i == 1 or i == 2: m=1
    if i == 3: m=[]
    if i == 4 or i == 5: m=2
    return m

def plot_block_and_cube(orig, last, mask, coords_Z, orig_small, coords_Z_small, last_small, mask_small):
    fig, ax = plt.subplots(2, 3, figsize=(7,5))
    ax[0,0].imshow(orig[coords_Z])
    ax[0,0].set_title(f'({idx}){i}, {n_ndl}')
    ax[0,0].axis('on')
    ax[0,1].imshow(last[coords_Z])
    ax[0,1].axis('on')
    ax[0,2].imshow(mask[coords_Z])
    ax[0,2].axis('on')
    ax[1,0].imshow(orig_small[coords_Z_small])
    ax[1,0].axis('on')
    ax[1,1].imshow(last_small[coords_Z_small])
    ax[1,1].axis('on')
    ax[1,2].imshow(mask_small[coords_Z_small])
    ax[1,2].axis('on')
    fig.tight_layout()
def plot_cube3(orig, last, mask, coords_Z, orig_small, coords_Z_small, last_small, mask_small):
    fig, ax = plt.subplots(1, 3, figsize=(9,3))
    ax[0].imshow(orig_small[coords_Z_small])
    ax[0].set_title(f'({idx}){i}, {n_ndl}')
    ax[0].axis('off')
    ax[1].imshow(last_small[coords_Z_small])
    ax[1].axis('off')
    ax[2].imshow(mask_small[coords_Z_small])
    ax[2].axis('off')
    fig.tight_layout()

def compare_labeled_and_df_coords(mask, coords_Z, coords_X, coords_Y, diff_thresh = 12):
    '''Compute the coords of each nodule in the mask and if they are close to the coords in the DF return them'''
    labeled, n_items = ndimage.label(mask)
    for i in np.arange(1,n_items+1):
        z,x,y = np.where(labeled==i)
        zz = int(np.median(z))
        xx = int(np.median(x))
        yy = int(np.median(y))
        if np.abs(coords_X - xx) < diff_thresh and np.abs(coords_Y - yy) < diff_thresh:
            minZ = min(z)
            maxZ = max(z)
            minX = min(x)
            maxX = max(x)
            minY = min(y)
            maxY = max(y)
            min_coords = [minZ, maxZ, minX, maxX, minY, maxY]
            return min_coords


# ## Main
path_data = '/data/OMM/Datasets/LIDC_other_formats/LIDC_preprocessed_3D v5 - save pylidc chars only/v17v3/'
path_chars =  f'{path_data}pylidc_characteristics/'
path_last = f'{path_data}arrays/last/'
path_orig = f'{path_data}arrays/orig/'
path_mask = f'{path_data}arrays/masks nodules/'
path_mask_lungs = f'{path_data}arrays/masks lungs/'
path_dest = '/data/OMM/Datasets/LIDC_other_formats/LIDC_inpainted_malignancy_classification v2/'

# files DIP reconstruction
files_last = os.listdir(path_last)
files_last = np.sort(files_last)
# files pylidc characteristics
files_chars = os.listdir(path_chars)
files_chars = np.sort(files_chars)
# files nodules masks
files_mask = os.listdir(path_mask)
files_mask = np.sort(files_mask)

# Get the files that are common to the DIP reconstruction and the pylidc characteristics
files_last_cropped = [i.split('.npy')[0] for i in files_last]
files_chars_cropped = [i.split('.csv')[0] for i in files_chars]
files_last_cropped = list(np.unique(files_last_cropped))

files_common = list(set(files_last_cropped).intersection(set(files_chars_cropped)))
files_common = np.sort(files_common)


# ## Continue
names_to_save = []
malignancies_original = []
malignancies = []
nodules_with_coords_errors = []
malignancies_mode, malignancies_mode_3_agree = [], []
names_to_save, names_to_save_3_agree = [], []

for idx, i in tqdm(enumerate(files_common), total=len(files_common)):
#     i_test = 2
#     if idx < i_test: continue
#     if idx >= i_test + 1: break
    
    # Get the inpainted and original image and the mask
    try:
        last = np.load(f'{path_last}{i}.npy')
        last = np.squeeze(last)
        orig = np.load(f'{path_orig}{i}.npy')
        orig = np.squeeze(orig)
        mask = np.load(f'{path_mask}{i}.npz')
        mask = mask.f.arr_0
        mask_lungs = np.load(f'{path_mask_lungs}{i}.npz')
        mask_lungs = mask_lungs.f.arr_0
    except FileNotFoundError: continue
            
    df = pd.read_csv(f'{path_chars}{i}.csv')
    n_nodules = np.unique(df['cluster_id'].values)
    for n_ndl in n_nodules: # for each nodule in the DF

        df_one_nodule = df.loc[df['cluster_id'] == n_ndl]
        coords_Z = int(np.mean(df_one_nodule['small_coordsZ_resampled'].values))
        coords_X = int(np.mean(df_one_nodule['small_coordsX'].values))
        coords_Y = int(np.mean(df_one_nodule['small_coordsY'].values))
        # if the DF and mask coords match then use the min and max of the latter 
        coords_limit = compare_labeled_and_df_coords(mask, coords_Z, coords_X, coords_Y)
        if coords_limit == None:
            nodules_with_coords_errors.append(f'{i}_{n_ndl}')
            continue
            
        # Get a cube around the nodule and the mask
        z_min_f, z_max_f, x_min_f, x_max_f, y_min_f, y_max_f = box_with_masks_search(coords_Z, coords_X, coords_Y, mask_lungs, coords_limit)
#         print(z_min_f, z_max_f, x_min_f, x_max_f, y_min_f, y_max_f)
        if z_min_f == -1:
            nodules_with_coords_errors.append(f'{i}_{n_ndl}')
            continue
        orig_small = orig[z_min_f: z_max_f, x_min_f:x_max_f, y_min_f:y_max_f]
        last_small = last[z_min_f: z_max_f, x_min_f:x_max_f, y_min_f:y_max_f]
        mask_small = mask[z_min_f: z_max_f, x_min_f:x_max_f, y_min_f:y_max_f]
        mask_lungs_small = mask_lungs[z_min_f: z_max_f, x_min_f:x_max_f, y_min_f:y_max_f]
        # Using the coords of the cube, get the coords of the nodule inside the cube
        if np.shape(orig_small) != (64,64,64):
            nodules_with_shape_errors.append(f'{i}_{n_ndl}')
            continue
            
        # Get the malignancy score
        malignancy1 = df_one_nodule.malignancy.values
        malignancy = list(map(transform_malignancy, malignancy1))
        malignancy = list(filter(None, malignancy))
        try:
            malignancy_mode = mode(malignancy)
        except StatisticsError: continue
            
        malignancies_original.append(malignancy1)
        malignancies.append(malignancy)
        malignancies_mode.append(malignancy_mode)
        # Next lines are to append to malignancies_mode_3_agree (if at least 3 reviewers agree on malignancy)
        agree_with_mode = [1 if malignancy_mode == i else 0 for i in malignancy]
        agree_with_mode = np.sum(agree_with_mode)
        if agree_with_mode >= 3:
            malignancies_mode_3_agree.append(malignancy_mode)
            names_to_save_3_agree.append(f'{i}_{n_ndl}')
        
        # Save figures and targets
        np.save(f'{path_dest}original/{i}_{n_ndl}.npy',orig_small)
        np.save(f'{path_dest}inpainted/{i}_{n_ndl}.npy',last_small)
        np.savez_compressed(f'{path_dest}mask/{i}_{n_ndl}',mask_small)
        names_to_save.append(f'{i}_{n_ndl}')
            
        # These coords can be used to 'plot_block_and_cube'
#         coords_Z_small = coords_Z - z_min_f 
#         coords_X_small = coords_X - x_min_f 
#         coords_Y_small = coords_Y - y_min_f 
#         plot_block_and_cube(orig, last, mask, coords_Z, orig_small, coords_Z_small, last_small, mask_small)

df_to_classify = pd.DataFrame.from_dict({'names': names_to_save, 'malignancy': malignancies_mode})
df_to_classify_3_agree = pd.DataFrame.from_dict({'names': names_to_save_3_agree, 'malignancy': malignancies_mode_3_agree})
df_to_classify.to_csv(f'{path_dest}df_classify_inpainted_malignancy.csv', index=False)
df_to_classify_3_agree.to_csv(f'{path_dest}df_3_agree_classify_inpainted_malignancy.csv', index=False)

