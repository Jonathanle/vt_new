#TODO: Transform the pd dataset to get the composite outcome


# Evalluate how to structure dataset such that we can always have a binary mask Create a binary mask

import torch
import pandas as pd
import os
import numpy as np 

#TODO - figure out a way to standardize the slice requirements for having a standard dataset. 

def create_data_tensors(): 
    """
    Which way is the best such that it leads to all the empirical outcomes that I need?
    
    How am i accessing data in training? 
    How can I have tools to analyze how the dataset works? 
    for anny patient dataset, I want to have the patient id.
    but I would also want to have a way of indexing by the idx of the tensor?o

    given an index i want to find the patient id
    given a patient_id i wnat to find the index? -- irrelevant i just want to find the elemnt thata is associated 

    if i had a way of having idx --> thing, then i just need to have a patient mapping --> idx --> thing? 
    seems very categorically intuitve no? 

    wrapper method? overloaded method?
    - always access stuff by using the ids; then we can use patient_id to have the same strucutre in sharing the dataset
    - intuitively we have at the interface the same process that allwos us to access the dataset elemetns


    """

    return
"""
TODO: Creaate the myocaardium mask

# input - raw mask
## lge mask - annotated 
- labels.

- create the label tensors and the verificaatino via pitest

"""

LABELS_FILEPATH = os.environ.get('LABELS_FILEPATH')



def create_outcome(filepath):
        """
        Returns a dictionary between the patient id composite outcome mappings
        
        Parameters:
        filepath (str): Path to the Excel file
        
        Returns:
        dict: Dictionary mapping patient IDs to their outcomes

        Assumptions - I decided to keep the variables static here because these explicitly are for the application of examining this pd file and its columns
        """
        # Read the Excel file
        df = pd.read_excel(filepath)
        

        # create a column composite
        df['composite'] = df['sustainedvt'] |  df['cardiacarrest'] |  df['scd'] 

        # Create a dictionary mapping Patient ID to Outcome
        outcome_dict = dict(zip(df['studyid'], df['composite']))
        
        return outcome_dict
def organize_patient_data(filepath):
    """
    Organizes patient data from a directory structure where each patient has their own
    numbered subdirectory containing .npy files.
    
    Args:
        filepath (str): Path to the main directory containing patient subdirectories
        
    Returns:
        dict: Dictionary where keys are patient IDs and values are dictionaries with 
              stacked tensors for 'cine', 'cine_whole', 'lge', and 'raw' data
    """
    data = {}
    
    # List all patient directories
    patient_dirs = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    
    for patient_id in patient_dirs:
        patient_path = os.path.join(filepath, patient_id)
        
        # Initialize dictionary for this patient
        patient_data = {
            'cine': None,
            'cine_whole': None,
            'lge': None,
            'raw': None
        }
        
        # Group files by category
        file_categories = {
            'cine': [],
            'cine_whole': [],
            'lge': [],
            'raw': []
        }
        
        # Get all .npy files in patient directory
        for filename in os.listdir(patient_path):
            if filename.endswith('.npy'):
                # More precise matching for categories
                if filename.startswith('cine_whole_'):
                    file_categories['cine_whole'].append(filename)
                elif filename.startswith('cine_'):
                    file_categories['cine'].append(filename)
                elif filename.startswith('lge_'):
                    file_categories['lge'].append(filename)
                elif filename.startswith('raw_'):
                    file_categories['raw'].append(filename)
        
        # Sort files to ensure correct slice order
        for category in file_categories:
            file_categories[category].sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        # Load and stack data for each category
        for category in file_categories:
            if file_categories[category]:
                # Load all files for this category
                loaded_files = []
                
                for filename in file_categories[category]:
                    file_path = os.path.join(patient_path, filename)
                    loaded_files.append(np.load(file_path))
                
                # Different handling based on category
                if category == 'cine_whole':
                    # For cine_whole, accept any shape but try to stack if consistent
                    shapes = [arr.shape for arr in loaded_files]
                    
                    if len(set(shapes)) == 1:
                        # All shapes are the same, stack them
                        stacked_tensor = np.stack(loaded_files, axis=0)
                        patient_data[category] = stacked_tensor
                    else:
                        # Inconsistent shapes - keep as a list for cine_whole
                        patient_data[category] = loaded_files
                else:
                    # For other categories, require consistent shapes
                    shapes = [arr.shape for arr in loaded_files]
                    if len(set(shapes)) == 1:
                        # All shapes are the same, stack them
                        stacked_tensor = np.stack(loaded_files, axis=0)
                        patient_data[category] = stacked_tensor
                    else:
                        # Inconsistent shapes for regular categories - try to find most common
                        from collections import Counter
                        shape_counts = Counter(shapes)
                        most_common_shape = shape_counts.most_common(1)[0][0]
                        
                        # Filter for files with the most common shape
                        consistent_files = [arr for arr in loaded_files if arr.shape == most_common_shape]
                        if consistent_files:
                            patient_data[category] = np.stack(consistent_files, axis=0)
                        else:
                            patient_data[category] = None
        
        # Add patient data to main dictionary
        data[int(patient_id)] = patient_data
    
    return data
def create_label_tensors():
    """
    Assemble dictionaries made into a larger dataset
    """
    outcome_dict = create_outcome(LABELS_FILEPATH)


    processed_data_dir = os.environ.get('PROCESSED_DATA_DIR')
    patient_data = organize_patient_data(processed_data_dir)


    final_dataset_dict = {}

    for patient_id, label in outcome_dict.items():

        # warning patient_id is a str. 
        if patient_id not in patient_data: 
            print(f'Warning: {patient_id} not in processed data skipping')
            continue

        label = label
        images = patient_data[patient_id]
    
        final_dataset_dict[patient_id] = {'label': label, 'images': images}
    

    return final_dataset_dict

class LGEDataset():
    """
    Dataset class that interfaces with a dictionary of cardiac MRI data.
    
    Input dataset_dict structure:
    {
        patient_id_1: {
            'label': int (0 or 1),
            'images': {
                'cine': np.ndarray (sliceno 2D, float32/float64), # Kept as numpy array
                'lge': np.ndarray (sliceno 2d, float32/float64),
                'cine_whole': np.ndarray (slicesno, 2D, float32/float64), # Kept as numpy array
                'raw': np.ndarray (sliceno 2D, float32/float64)
            }
        },
        patient_id_2: {...},
        ...
    }
    
    Returns:
    {
        'patient_id': str,
        'label': torch.tensor (shape [], dtype=torch.long),
        'images': {
            'cine': np.ndarray (2D, float32/float64), # Original numpy array
            'lge': torch.tensor (shape [1, slice_no, H, W], dtype=torch.float32),
            'cine_whole': np.ndarray (2D, float32/float64), # Original numpy array
            'raw': torch.tensor (shape [1, slice_no, H, W], dtype=torch.float32)
            'myomask': torch.tensor (shape [1, slice_no, H, W], dtype = torch.float32
        }
    }
    """
    # TODO: EXAMINe what datatype myomask outputs


    def __init__(self, dataset_dict): 
        """
        Initialize the dataset with a dictionary of patient data.
        
        Args:
            dataset_dict: Dictionary with patient data structured as specified in the class docstring.
            'cine' and 'cine_whole' are kept as numpy arrays since they won't be used in training.
        """
        self.dataset_dict = dataset_dict
        
        # Create a mapping from patient ID to index
        self.patient_ids = list(dataset_dict.keys())
        self.id_to_idx = {patient_id: idx for idx, patient_id in enumerate(self.patient_ids)}
        
        # Convert only necessary numpy arrays to PyTorch tensors
        for patient_id, data in self.dataset_dict.items():
            # Convert label to tensor
            data['label'] = torch.tensor(data['label'], dtype=torch.long)
           
            # TODO: Unverified work with this + document with cost the 
            # I verified this simply using a matplotlib visualize
            data['images']['myomask'] = data['images']['raw'] > 1e-3

            # Convert only specified images to tensors with channel dimension
            for img_type, img_array in data['images'].items():
                # Skip cine and cine_whole - keep as numpy arrays
                if img_type in ['cine', 'cine_whole']:
                    continue
                    
                # Add channel dimension if not present (relevant for depth) of images
                if img_array.ndim == 3:
                    img_array = np.expand_dims(img_array, axis=0)  # [1, slice, H, W]
                
                # Convert to float32 tensor
                data['images'][img_type] = torch.tensor(img_array, dtype=torch.float32)
    
    def __len__(self):
        return len(self.dataset_dict)
    
    def __getitem__(self, idx): 
        """
        Fucntion that retrieves a dataset based on the idx or the dataset

        Uses specifically the datatype to disambiguuate the method for accessing



        Returns dict: {'patient_id': (id), 'label': torch.Tensor(), 'dataset': 0} 


        dataset = {'cine': torch dataset , torch. numpy
        """
        # Get the patient ID for this index
        if isinstance(idx, int): 
            patient_id = self.patient_ids[idx]
        
        elif isinstance(idx, str): 
            patient_id = int(idx)  # use str to then inform of the idness
        else: 
            raise Exception("error idx is not an int or str)")
        


        # Get the data for this patient
        data = self.dataset_dict[patient_id]
        
        # Return the data along with the patient ID
        return {
            'patient_id': patient_id,
            'label': data['label'],
            'images': data['images']
        }

def main(): 

    dataset_dict = create_label_tensors()
    dataset = LGEDataset(dataset_dict)

    breakpoint()

if __name__ == '__main__':
    main()
