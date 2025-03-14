"""
Function for generating ground truth segmentations for the segmentation


Directory Format 
- Must in the directory structure contain 'raw' to process

- PATIENTDIR \ patient_files


"""


from .visualizer import get_data_directory, get_patient_directories


from attention_model import AttentionUNet # will import becauase i run from the root directory
# running python3 from rooot will import the attention model (no need for adding anything)

import torch 
import os 
import numpy as np


DEVICE = 'cuda:0' # TODO: sb and see ways I can formalize into a configuration file for best organization


import re
from pathlib import Path

def create_lge_files(patient_dir, model):
    """
    Create the ground truth LGE files for visualization using the Swin U-Net Model as ground truth.

    Input patient_dir (PosixPath) - Path to the patient directory where we then explore

    - regex + 
    """
    # DF - How do i work with directories? 
    # specifically i want to look in a patient_dir files, find files thaat satisfy a certain type of string, 
    # then for each do something with it (np load)
    # then for each file save it into another file --> seems very important for data relaated skills
    # todo - reduce and formalize the goals related to "working with directories" and manipulating strings using raw as reference

    # Goal - Get all the Paths that satisfy the string configuration "raw"
    pattern = re.compile(r'raw')

    # iteratethrouth dir then use re to evaluate - 'is there a match?' -- TODO: refactor this into insights learned later
    raw_matching_items = [item for item in patient_dir.iterdir() if pattern.search(item.name)]



    # validate to reject any directories that have any* lge files as this makes it complex
    pattern = re.compile('lge')
    lge_matching_items = [item for item in patient_dir.iterdir() if pattern.search(item.name)]

    assert len(lge_matching_items) == 0


    # create the output file 
    for raw_filepath in raw_matching_items:

        data = torch.tensor(np.load(raw_filepath), dtype=torch.float32).unsqueeze(0).unsqueeze(0) # align with pytorch to have shape (batch, channel, height, width
        
        data = data.to(DEVICE)
        
        result = model(data)# TODO: Document for Personal Reasons the datatype of the thing through testing double formatiokn
        # Model must take only float32 not float64
        # get the numpy version by detaching? why impt - getting to cpu (helps to work in nupy domain.
        result_numpy = result.squeeze(0).squeeze(0).detach().cpu().numpy()
       

        # build the file to save
        # get the integer
        # how to find and extract specific text from a text string (given text string how to find the outside)
        match = re.compile(r'raw_(\d+)\.npy$')
        match1 = re.search(r'raw_(\d+)\.npy', str(raw_filepath)) # find the greedy first match could complicate if the filenams are 'raw' and such
        # get the digit group out
        slice_num = int(match1.group(1)) # arbritrary: re uses 1 indexing 0 = entire matched string that you want use this as example

        lge_filename = f'lge_{slice_num}.npy'
        np.save(os.path.join(patient_dir, lge_filename), result_numpy)



    return


def remove_lge_files(patient_dirs): 
    """
    Cleans all lge_files all patient directories according to if re.searach returns True or not.

    Paraams: 
        patient_dirs list[Posix Path] - list containing the posix paths


    """

    #TODO: Test as needed as I definitely believe that there are bugs



    for patient_id, patient_dir in patient_dirs.items(): 
        # search for all files that contain LGE
        pattern = re.compile(r'lge')
        
        lge_matching_items = [item for item in patient_dir.iterdir() if pattern.search(item.name)] # TODO: reorganize importance so thata it is useful
# re.saerch return a match or None, which then can be interprted as a match or nto

        for lge_file in lge_matching_items:
            os.remove(os.path.join(patient_dir, lge_file))


        # for each of these filepaths do something to remove it 

def main(): 
    data_dir = get_data_directory()
    patient_dirs = get_patient_directories(data_dir)
    
    root_directory = os.environ.get('PROJECT_ROOT')

    remove_lge_files(patient_dirs)

    model_path = os.path.join(root_directory, 'models', 'unet_att_focal_dice350.pt')

    model = AttentionUNet(drop_out_prob=0.3).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for patient_id, patient_dir in patient_dirs.items(): 
        create_lge_files(patient_dir, model)


    

if __name__ == '__main__':
    main()
