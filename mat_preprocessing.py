"""
MAT Preprocessing for Cardiac MRI Analysis

High-Level Description:
This script processes cardiac MRI data stored in MATLAB (.mat) files, focusing on Late Gadolinium 
Enhancement (LGE) sequences used to identify myocardial scarring. The script extracts the 
myocardium region using provided contour data, applies standardization to the images, 
and outputs cropped and processed numpy arrays ready for deep learning analysis.

Inputs:
- Directory of MATLAB files (default: './data/Matlab/')
 Format:
 ./data/Matlab/
 ├── patient_id_1/
 │   └── patient_id_1_PSIR.mat (or _LGE.mat or _MAG.mat)
 ├── patient_id_2/
 │   └── patient_id_2_PSIR.mat
 └── ...

- MATLAB File Contents:
 - series_type: Must be "Myocardial Evaluation"
 - raw_image: Raw MRI scan data array
 - lv_endo: Left ventricle endocardium contour coordinates
 - lv_epi: Left ventricle epicardium contour coordinates
 - enhancement: Data showing areas of enhancement (scarring)

Outputs:
- Processed numpy files (saved to './data/preprocessed_files/' by default)
 Format:
 ./preprocessed_files/
 ├── patient_id_1/
 │   ├── raw_0.npy        # Slice 0: Myocardium-only masked image
 │   ├── cine_0.npy       # Slice 0: Cropped raw image
 │   ├── cine_whole_0.npy # Slice 0: Full uncropped image
 │   ├── lge_0.npy        # Slice 0: LGE segmentation (scarring)
 │   └── ... (additional slices)
 └── ...

Note: The bottom section of this script contains code for collecting all processed 
data into a JSON file. This is supplementary functionality that may be removed if 
not needed for your analysis pipeline.
"""


"""
Summary of modifications made from the original jupyter notebook to this file 

1. Transformed this into a script
2. Enabled saving of images to non LGE forma
3. Added CLI Interface for more flexible preprocessing

"""

# ##### Import all libraries

import scipy.io as sio
import scipy
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import tomni
from matplotlib import cm
from sklearn.model_selection import train_test_split
import torch

import argparse




root_directory = os.environ.get('DATA_DIR')

input_directory =  os.environ.get('UNPROCESSED_DATA_DIR')#os.path.join(root_directory, 'Matlab')
output_directory = os.environ.get('PROCESSED_DATA_DIR')#os.path.join(root_directory, 'preprocessed_files')


# IDEA[1]: Parse Args is something I dont need to do and is in this context a 'calculator task'
def parse_args():
    parser = argparse.ArgumentParser(
        description='Process cardiac MRI MATLAB files to extract myocardium regions and standardize images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # TODO[0]: Change dependency for preprocessing file to be strictly within a specific file directory    
    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        required=False,
        default=input_directory, #IDEA[0]: Syntax, I learned to better specify explicitly a directory (think directory/)
        help='Directory containing patient MATLAB (.mat) files with LGE/PSIR/MAG sequences'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=False,
        default=output_directory,
        help='Directory where processed numpy files will be saved'
    )
    parser.add_argument(
        '-s', '--size', 
        type=int, 
        required=False,
        default=128, 
        help='Dimensions of the resulting mat file images'
    )
   
    return parser.parse_args()

args = parse_args()

size = args.size
# ##### This snippet gets the list of .mat files under each subject


path= args.input_dir #args.input_dir#TODO: Refactor this to get into command line
output_dir = args.output_dir

subjects= os.listdir(path)
for si in range(len(subjects)):
    new_path= os.path.join(path,subjects[si]) #EDIT[0]: Changed this from path+subjects for flexible parsing
    files= os.listdir(new_path)
print('Total number of subjects: ', len(subjects))
print(files)


''' Sample: loading the last subject's PSIR file to check its keys '''
flge= sio.loadmat(new_path + '/'+ files[0]) # modified thtis to look at files 0 (originally buggy from 2)
flge.keys()

# #### Saving Cropped Images


''' Standardization and resizing functions '''

def std_img(tens):
    t_ = (tens-np.amin(tens))/(np.amax(tens)-np.amin(tens))
    return t_

# TODO: Include option to change the image size
def resize_volume(img, ex=size):              ### THIS IS CURRENTLY OPERATING ON 2D DATA, 'ex' IS THE EXPECTED SIZE
    current_depth = img.shape[0]
    current_width = img.shape[1]            

    depth_factor = ex / current_depth
    width_factor = ex / current_width

    factors = (depth_factor, width_factor)

    return scipy.ndimage.zoom(img, factors, order=1)


''' Initialize all variables to start saving '''

path= './data/Matlab/' # TODO: Find out if this is useful or not
subjects= os.listdir(path)
si = 0
yes_save = 1



def saver(si):
    saving_folder = output_dir 
    os.makedirs(saving_folder, exist_ok=True)
    if yes_save:
        try:
            os.mkdir(os.path.join(saving_folder, subjects[si]))
        except:
            print('error creating folder; folder already exists: ', os.path.exists(saving_folder +subjects[si]))

    new_path= path+subjects[si]
    files= os.listdir(new_path)
    for fs in files:
        # slicer= (75, 175, 45, 145)  ### if you want to manually crop the images
        if 'PSIR' in fs or 'LGE' in fs or 'MAG' in fs:
            flge= sio.loadmat(new_path+'/'+fs)

            try:
                __ = flge['series_type']
            except:
                print('no series for ', fs)
                continue
            assert flge['series_type']==np.array(['Myocardial Evaluation'])

            print(('Length: ', flge['enhancement'][0].shape[0]))

            for slice_no in range(flge['enhancement'][0].shape[0]):
                scar= np.copy(flge['enhancement'][0][slice_no]).astype('float')
                scar[scar==0]=np.nan
                if True: #EDIT - changed this from np.nansum(scar) !=0 to allow for scars 
                # if 1:
                    try:
                        _= flge['lv_endo'][0][slice_no][0][0]
                    except:
                        print('couldnt get lv_endo')
                        continue
                    #EDIT - added a validation check #
                    #WHYIMPT - Some files only contained a LV or EPI contour --> need changes
                    try:
                        _= flge['lv_epi'][0][slice_no][0][0]
                    except:
                        print('couldnt get lv_epi skipping')
                        continue

                    """
                    Rotation Section Added
                    """
                            

                    # flge['rv_insertion'][0][slice_no][0][0][0][0].shape 
                     # Right before the mask creation:
                    # Access RV insertion point

                    try:
                        rv_insertion = flge['rv_insertion'][0][slice_no][0][0][0][0]  # Shape (2,) with [x, y] coordinates

                    except:
                        print("couldn't get the rv insertion point")
                        continue


                    # Find LV center from endocardial contour
                    lv_contour = flge['lv_endo'][0][slice_no][0][0][0]
                    lv_center = np.mean(lv_contour, axis=0)  # [x, y] format

                    # Calculate current angle of RV insertion relative to center
                    current_angle = np.arctan2(rv_insertion[1] - lv_center[1], 
                                              rv_insertion[0] - lv_center[0])

                    # Define target angle (e.g., 90 degrees or π/2 radians)
                    target_angle = np.pi/2  # Or any other standardized angle
                    rotation_angle = (target_angle - current_angle) * -1 # empimrically realized an anti rotation  
                    rotation_angle_degrees = np.degrees(rotation_angle)

                    # Import necessary functions
                    from scipy.ndimage import rotate, shift

                    # For the image, we need a special approach to rotate around a specific center:
                    raw_image = flge['raw_image'][0,slice_no]
                    img_center = np.array(raw_image.shape) / 2  # Center of image (y, x)
                    offset = img_center - np.array([lv_center[1], lv_center[0]])  # Offset from image center to LV center (y, x)

                    # 1. Shift to center the LV center
                    centered_image = shift(raw_image, offset)
                    # 2. Rotate around image center
                    rotated_image = rotate(centered_image, rotation_angle_degrees, reshape=False, order=1)
                    # 3. Shift back
                    rotated_raw_image = shift(rotated_image, -offset)

                    # Create rotation function for contours
                    def rotate_contour(points, center, angle_degrees):
                        """Rotate points around center by angle in degrees."""
                        angle_rad = np.radians(angle_degrees)
                        # Center the points
                        centered_points = points - center
                        # Create rotation matrix
                        cos_angle = np.cos(angle_rad)
                        sin_angle = np.sin(angle_rad)
                        rotation_matrix = np.array([
                            [cos_angle, -sin_angle],
                            [sin_angle, cos_angle]
                        ])
                        # Apply rotation and un-center
                        rotated_points = np.dot(centered_points, rotation_matrix.T) + center
                        return rotated_points

                    # Rotate contours
                    rotated_endo = rotate_contour(flge['lv_endo'][0][slice_no][0][0][0], lv_center, rotation_angle_degrees)
                    rotated_epi = rotate_contour(flge['lv_epi'][0][slice_no][0][0][0], lv_center, rotation_angle_degrees)
                   


                    # Replace originals with rotated versions
                    flge['raw_image'][0,slice_no] = rotated_raw_image
                    flge['lv_endo'][0][slice_no][0][0][0] = rotated_endo
                    flge['lv_epi'][0][slice_no][0][0][0] = rotated_epi

                    # Now proceed with your existing code - no changes needed
                    img_shape = np.transpose(flge['raw_image'][0,slice_no]).shape
                    myo_seg_endo = tomni.make_mask.make_mask_contour(img_shape, flge['lv_endo'][0][slice_no][0][0][0])
                    myo_seg_epi = tomni.make_mask.make_mask_contour(img_shape, flge['lv_epi'][0][slice_no][0][0][0])                   


                    def debug_plt_show():
                        """
                        Debug function to visualize original and rotated images with center points and RV insertion.
                        Call this function in pdb to compare the images before and after rotation.
                        """
                        import matplotlib.pyplot as plt
                        from copy import deepcopy
                        
                        # Create a figure with two subplots side by side
                        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                        
                        # Get the original image and contours (we need to store these before rotation)
                        # Make deep copies to ensure we don't modify the originals
                        original_image = deepcopy(raw_image)
                        original_endo = deepcopy(flge['lv_endo'][0][slice_no][0][0][0])
                        original_epi = deepcopy(flge['lv_epi'][0][slice_no][0][0][0])
                        original_center = np.mean(original_endo, axis=0)  # [x, y] format
                        
                        # Left subplot - Original image
                        axes[0].imshow(original_image, cmap='gray')
                        axes[0].plot(original_endo[:, 0], original_endo[:, 1], 'r-', linewidth=1)  # Endocardial contour
                        axes[0].plot(original_epi[:, 0], original_epi[:, 1], 'b-', linewidth=1)    # Epicardial contour
                        
                        # Plot center of mass and RV insertion point on original image
                        axes[0].plot(original_center[0], original_center[1], 'go', markersize=8)  # LV center as green dot
                        axes[0].plot(rv_insertion[0], rv_insertion[1], 'yo', markersize=8)        # RV insertion as yellow dot
                        
                        # Draw a line from LV center to RV insertion to visualize current angle
                        axes[0].plot([original_center[0], rv_insertion[0]], 
                                     [original_center[1], rv_insertion[1]], 'y-', linewidth=2)
                        
                        # Add title and angle info to original image
                        axes[0].set_title(f'Original Image\nRV Insertion Angle: {np.degrees(current_angle):.1f}°')
                        
                        # Right subplot - Rotated image
                        axes[1].imshow(rotated_raw_image, cmap='gray')
                        axes[1].plot(rotated_endo[:, 0], rotated_endo[:, 1], 'r-', linewidth=1)  # Rotated endocardial contour
                        axes[1].plot(rotated_epi[:, 0], rotated_epi[:, 1], 'b-', linewidth=1)    # Rotated epicardial contour
                        
                        # Get rotated center and plot it
                        rotated_center = np.mean(rotated_endo, axis=0)  # [x, y] format
                        axes[1].plot(rotated_center[0], rotated_center[1], 'go', markersize=8)  # LV center as green dot
                        
                        # Calculate the expected position of RV insertion after rotation
                        rotated_rv = rotate_contour(np.array([rv_insertion]), lv_center, rotation_angle_degrees)[0]
                        axes[1].plot(rotated_rv[0], rotated_rv[1], 'yo', markersize=8)  # Rotated RV insertion as yellow dot
                        
                        # Draw a line from rotated LV center to rotated RV insertion
                        axes[1].plot([rotated_center[0], rotated_rv[0]], 
                                     [rotated_center[1], rotated_rv[1]], 'y-', linewidth=2)
                        
                        # Calculate the new angle after rotation
                        new_angle = np.arctan2(rotated_rv[1] - rotated_center[1], 
                                              rotated_rv[0] - rotated_center[0])
                        
                        # Add title and angle info to rotated image
                        axes[1].set_title(f'(Deprecated) Rotated Image\nRV Insertion Angle: {np.degrees(new_angle):.1f}°\nTarget Angle: {np.degrees(target_angle):.1f}°')
                        
                        # Add legend
                        from matplotlib.lines import Line2D
                        legend_elements = [
                            Line2D([0], [0], color='r', lw=1, label='Endocardium'),
                            Line2D([0], [0], color='b', lw=1, label='Epicardium'),
                            Line2D([0], [0], marker='o', color='g', label='LV Center', markersize=8, linestyle='None'),
                            Line2D([0], [0], marker='o', color='y', label='RV Insertion', markersize=8, linestyle='None')
                        ]
                        fig.legend(handles=legend_elements, loc='lower center', ncol=4)
                        
                        # Adjust layout
                        plt.tight_layout()
                        plt.subplots_adjust(bottom=0.15)  # Make room for the legend
                        
                        # Show the plot
                        plt.show()

                    #12272 slice 0 - opposite rotation erro

                    if subjects[si] == '8687' and slice_no == 0:
                        continue
                        breakpoint() #TODO: verify that the rotation is property working andd no errors
                        #TODO: Document Why the rotation is working the opposite direction
                    



                    img_shape= np.transpose(flge['raw_image'][0,slice_no]).shape

                    myo_seg_endo= tomni.make_mask.make_mask_contour(img_shape, 
                                                                    flge['lv_endo'][0][slice_no][0][0][0])    ### CONVERT CONTOURS TO BINARY MASKS
                    myo_seg_epi= tomni.make_mask.make_mask_contour(img_shape,
                                                                flge['lv_epi'][0][slice_no][0][0][0])         ### CONVERT CONTOURS TO BINARY MASKS

                    # TODO: figure out a way to flip the contoour shape, or add data for encoding. 
                    myo_seg= (myo_seg_epi - myo_seg_endo).astype('float')
                    flge['raw_image'][0,slice_no]/=np.amax(flge['raw_image'][0,slice_no])
                    myo_seg[myo_seg==0]= np.nan
                    fin_img= flge['raw_image'][0,slice_no]*myo_seg
                    imc_ = flge['raw_image'][0,slice_no]
                    imc_full = std_img(np.array(flge['raw_image'][0,slice_no]))
                    fin_img[np.isnan(fin_img)]=0                                                              ### NEEDED TO SAVE


                    im= Image.fromarray(np.uint8(cm.gray(fin_img)*255)).convert('L')
                    imc__ = Image.fromarray(np.uint8(cm.gray(imc_)*255)).convert('L')
                    scar_im= Image.fromarray(np.uint8(cm.gray(scar)*255)).convert('L')

                    ''' Get bounding boxes from contours and crop the images. '''
                    im.getbbox()                                                                              
                    im2 = (std_img(np.array(im.crop(im.getbbox()))))   ## cropped raw image with myo only
                    im2 = resize_volume(im2)

                    imc = (std_img(np.array(imc__.crop(im.getbbox()))))  ## cropped raw image
                    imc = resize_volume(imc)

                    sc2 = (std_img(np.array(scar_im.crop(im.getbbox()))))   ## cropped lge segmentation
                    sc2 = resize_volume(sc2)



                    if yes_save:
                        
                        save_path= os.path.join(saving_folder, subjects[si], 'raw_'+ str(slice_no) + '.npy') # DF: why /raw is bad in text processing


                        im2[np.isnan(im2)]=0
                        np.save(save_path, im2)

                        save_path= os.path.join(saving_folder, subjects[si], 'cine_'+ str(slice_no) + '.npy')
                        np.save(save_path, imc)

                        save_path= os.path.join(saving_folder, subjects[si], 'cine_whole_'+ str(slice_no) + '.npy')
                        np.save(save_path, imc_full)

                        # WHYIM: Commented out to emphasize in the output no* lge output, make more consistent 
                        #save_path= os.path.join(saving_folder, subjects[si], 'lge_'+ str(slice_no) + '.npy')
                        #sc2[np.isnan(sc2)]=0
                        #np.save(save_path, sc2)



for si in range(len(subjects)):
    print(f"saving {si}")
    saver(si)

# ##### Intermediate .npy files used for train-test split, not for directly storing as JSON

# Code for loading into JSON

# WHYIM: commented out because I suspect with 80% belief that it is not relevant for the saving itself only for JSON 
#
#key_dir = output_dir 
#
#raw_=[]
#lge_=[]
#cine_=[]
#cine_whole_=[]
#
#for subject in os.listdir(key_dir):    
#    files= os.listdir(os.path.join(key_dir, subject))
#    for f in files:
#        idx= f[4:list(f).index('.')]
#        if 'raw' in f:
#            raw_.append(np.load(os.path.join(key_dir, subject, f)))
#
#            lge_f = 'lge_'+str(idx)+'.npy'
#            lge_.append(np.load(os.path.join(key_dir, subject, lge_f)))
#
#            lge_f = 'cine_'+str(idx)+'.npy'
#            cine_.append(np.load(os.path.join(key_dir, subject, lge_f)))
#
#            lge_f = 'cine_whole_'+str(idx)+'.npy'
#            cine_whole_.append(resize_volume(np.load(os.path.join(key_dir, subject, lge_f)), ex=224))
#
#raw_ = np.array(raw_)
#lge_= np.array(lge_)
#cine_= np.array(cine_)
#cine_whole_ = np.array(cine_whole_)
#
#raw_.shape, lge_.shape, cine_.shape, cine_whole_.shape
"""
# ##### Store as JSON


import json

datas = {'lge_whole': cine_whole_,
         'lge_cropped': cine_,
         'masked_input': raw_, 
         'lge_seg': lge_}

def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))

with open("./sample_inference_data.json", "w") as outfile: 
    json.dump(datas, outfile, default=default)


# TODO: Evaluate if this is a reasonaable objective
inf_data = json.load(open("./sample_inference_data.json"))
inf_data.keys()


lge_whole= np.array(inf_data['lge_whole']).reshape(44,224,224)
x = np.array(inf_data['masked_input']).reshape(44,64,64)
y = np.array(inf_data['lge_seg']).reshape(44,64,64)
cropped = np.array(inf_data['lge_cropped']).reshape(44,64,64)

"""
