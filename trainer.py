"""
TODO: bootstrap a prompt showing creating a baseline model

1


"""



"""
Refactored commends: 

*q1: should I consider adding the dimension or voltage or not? there is must 1 valule? added for consisteny of description

"lower temporal frame"? ---> what about a lower "contextual frame that I also defined? this seems to be the case with processes such as my standard temporal and contextual frame as encoding all of the most relevant ideas / being bery local? 

"""

import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from torch.optim import Adam
# import torch.nn.functional as F

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import pandas as pd
from pathlib import Path

import re
import inspect # function used for checking code structure at runtime


from dataset import LGEDataset, create_label_tensors
from model import VANet #TODO: Create the ECGNet


# TODO: Verify training workflow works with this


def collate_fn(batch):
    """
    Custom collate function that converts dictionary batch to tuple format
    for compatibility with existing training loop
    
    REMOVES THE KEYS AND ANY RELEVANT IDENTIFYING INFORMATION CHANGE LATER AS NEEDED
    """

    # TODO: Test functionality of this simply


    lge_images = [item['images']['lge'] for item in batch]
    myo_masks = [item['images']['myomask'] for item in batch]

    
    # this may make sense because what happens is that the images assume the same size in batch (they arent cross samples)
    # neeed to handle variable sizeed tensors (assumption of batch processing is not there anymore)
    #lge_images  = torch.stack([item['images']['lge'] for item in batch])
    #myo_masks = torch.stack([item['images']['myomask'] for item in batch])

    data = []
    for lge_image, myo_mask in zip(lge_images, myo_masks):
        data.append(torch.stack([lge_image, myo_mask], dim=1)) # the error is here, becaause I just create a single batch
        

    labels = torch.stack([item['label'] for item in batch]) 
    patient_ids = [item['patient_id'] for item in batch]
    


    #DF: Concern over the torch stack and list being heterogenous
    return data, labels, #patient_ids

def train_epoch(training_config, model, train_loader, criterion, optimizer, device):
    """
    Train one epoch - DFD does the model have to know about the configuration of the dataset? 
    Yes - It has to know specifically that any* dataset must be a tuple 
    - LT probably wasnt the best option for a dataset interface, but next time i can do that. change to dict?
    - well nothing changes, like there is always data, there is always labels, there is always IDs, 
   -> therefore any dataset MUST output a 3-tuple. 
    
    """
    model.train()
    running_loss = 0.0
    predictions = []
    true_labels = []
   
    
    for data, labels in train_loader: 
        """
        Custom Training Loop for handling variable sized data
        """
        
        # Question is the data training correctly? 
         
        labels = labels.to(device)
        outputs = torch.zeros_like(labels, dtype=torch.float32)# TODO: define how to make this


        optimizer.zero_grad()
        

        # Get the output for the batch sizes 
        for i, data_tensor in enumerate(data):
            data_tensor = data_tensor.to(device)
            output = model.forward(data_tensor)
            
            # TODO: get a binary prediction from output model

            outputs[i] = output # Are there any complexies i must ocnsider for shape (1,1) setting inside the hting? 

        labels = labels.float().to(device)  # Convert to float for BCE loss note for later + add to device
        optimizer.zero_grad()
        

        loss = criterion(outputs, labels) # TODO: analyze and formalize behavior of test in this in the test what exactly need the inputs to be? )
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predictions.extend(outputs.detach().cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader)
    epoch_auc = roc_auc_score(true_labels, predictions)
    
    return epoch_loss, epoch_auc

class TrainingConfig():
    """
    Class That handles Configuration Management and Instantiation

    Particular Important for handling the mapping and just the handlinlg of creation
    we just wnat the code to just do the doing of making the model, and not having it need to define its own configuraation
    or even validate that the configuration is useful

    Then Training handles telling the functions training what to do and the "env" of training. 
    It will validate different combinations of patient_datasets to the models.

    Acts as a configuration Objective Variable for telling what to do / The ending state + encoding dataset env.
    """
    # TODO: change to add model configurations + default model 
    def __init__(self, dataset = LGEDataset, model = VANet): 
        """
        Initialize Parameters

        Args: 
            ECGDataset (Dataset Class) - name of the class to instantiate

        """         
        self.num_epochs = 100
        self.patience = 10 
        

        self.dataset_model_configurations = {(LGEDataset, VANet)} # Add more valid Dataset, Model Configurations as needed (this is manually added knowledge by me)

        
        self.dataset = dataset# why are these models in params vs inside?
        self.model = model 

        # Validation check for dataset and model to be *types* and not classes
        assert isinstance(self.dataset, type) # classes are "type" NOT the clas
        assert isinstance(self.model, type)
        

        self.validate_dataset_model_config(self.dataset, self.model) # always use called in functions to emphasize existence watch out for hidden inputs from class


        self.use_post = False # TODO: for every subjective uncertainty - I will verify with testing does this actuallly go to post (manual verifcation says yes)
        self.class_weights = torch.Tensor([0.8])#.to(device) removed to becauase .to is a device specific function)
        self.criterion = torch.nn.BCELoss(weight=self.class_weights)  
        self.lr = 0.001

        # optimizer has an instantiation dependency to make model --> when creating model i can then create optimizer
        self.optimizer_class = torch.optim.Adam #(model.parameters(), lr=self.lr) # TODO: instantiate Adam when model is instantiated
        self.optimizer = None
    
        # Dataloader configurations
        self.batch_size = 8
        self.n_workers = 0  


        # CV Configurations
        self.n_splits = 5

        self.device = 'cuda:0'
    def validate_dataset_interface(self):
        """
        Heuristic Check on idx 0 that the number of elements is 3 and fulfills properties of data

        Requirements checked: 
            1. data must be a Pytorch Tensor
            2. labels must be Pytorch and Binary Integer Outcome
            3. Last must be a string of the form RXX

        Create Dataset? 
        # no i want to check the function signature and the return
        """
            
        #sig = inspect.signature(self.dataset.__getitem__)
        # TODO: Implement Validation Interface for faster type checking 
            
        return
    def validate_dataset_model_config(self, dataset, model):
        """
        Training Config Checks to see if the Dataset is good before telling what to do.
        Checks if the configuration is consistent and found in the valid configurations
        """
        return (dataset, model) in self.dataset_model_configurations
    
    def get_dataset(self):
        """
        Using VA, gets the dataset

        Params (Implicit): 
            self.post (boolean) - boolean must be either true or false
        """ 
    
        dataset_dict = create_label_tensors()
        dataset = LGEDataset(dataset_dict)


        return dataset

    def generate_model(self): 
        """
        Returns a dictionary containing 4 attributes for training based off of the training config
        """

        model = self.model()
        self.optimizer = self.optimizer_class(model.parameters(), lr=self.lr)

        return model
    def generate_dataloader(self, dataset, split): 
        """
        Generates a dataloader that from indices - (should the indices here be inputted into the function? how does Dataset relate to Config? Should it be controling dataset or fed? 
        """

        # TODO: fit the VA dataset to retrieve the relevant tensors --> create another dataset
        dataloader = DataLoader(dataset, 
            batch_size=self.batch_size, 
            sampler=SubsetRandomSampler(split), # notice here that the train variable greatly simplifies the approach (pseudo discovery based)
            collate_fn=collate_fn,
            num_workers=0
        )

        return dataloader  
    

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    true_labels = []
   
    # TODO[0]:
    with torch.no_grad():


        for data, labels in val_loader:
         
            labels = labels.to(device)
            labels = labels.float()

            outputs = torch.zeros_like(labels, dtype=torch.float32)# TODO: define how to make this

            # Get the output for the batch sizes 
            for i, data_tensor in enumerate(data):
                data_tensor = data_tensor.to(device)
                output = model.forward(data_tensor)
                
                # TODO: get a binary prediction from output model

                outputs[i] = output # Are there any complexies i must ocnsider for shape (1,1) setting inside the hting? 


            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(val_loader)
    val_auc = roc_auc_score(true_labels, predictions)
    
    return val_loss, val_auc

def train_model(training_config, model, optimizer, criterion, train_loader, val_loader, device, save_model=True):
    """
    Train a Model To optize towards parameters

    Params:
        model (torch.nn) - model for optimizing
        optimizer (torch.optim) - model for optimizing
        criterion (torch.nn.modules.loss) - 
        train_loader (torch.??) - Datalloader for training data
        val_loader (torch.??) - Dataloader for validation data
        save_model (boolean) - flag to save the best AUC model. Defaults to True

    Returns: (what? I need to be very intentional about how I create the logic)o

    Side Effects: 
        - This will Report Current Training Performance in Epochs to stdout
        - Will write the best modell to /model to torch.state_dict with the follwoing format:
'epoch': epoch,                                                                         - 
    TODO: Get More Description and Organize the Architecture to prevent entropy 
    

    """
    # TODO: Refactor into a Main Config Class #TrainConfig
    num_epochs = training_config.num_epochs 
    best_val_auc = 0
    best_model_state = None
    patience = training_config.patience# this is something thaat one defines what do to 
    patience_counter = 0
    # load model parameters for validation 

    for epoch in range(num_epochs):
        # Train
        train_loss, train_auc = train_epoch( 
            training_config,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        # Validate
        val_loss, val_auc = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )

        # Print metrics
        print(f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')

    
        # Model selection and early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0

            if save_model == False:
                continue

            # Save best model
            save_dir = Path('models')
            save_dir.mkdir(exist_ok=True)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auc': val_auc,
            }, save_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered')
                break

    return best_val_auc



def create_data_splits(dataset, n_splits=5, val_size=0.2):
    """
    Create train/val/test splits that can be extended to k-fold CV.
    Initially uses only one fold but structured for easy k-fold extension.
    
    For dataset only requires informataion about dataset's size AND binary outcome to have equal distribution   
    Args:
        dataset: ECGDataset instance
        n_splits: Number of folds (default=5)
        fold_idx: Which fold to use as test set (default=0)
        val_size: Size of validation set as fraction of training data
    
    Returns:
        list of folds with a dict containing train, val, and test indices
    """
    # Get labels for stratification + splitting
    labels = []
    for i in range(len(dataset)):
        _, label, _ = dataset[i] #NOTE coupled assumption about dataset structure

        #EDIT[0]: Replaced with label

        label = dataset[i]['label']

        labels.append(label.item())
    labels = np.array(labels)
    
    # Create stratified k-fold splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Get indices for the specified fold
    splits = list(skf.split(np.zeros(len(dataset)), labels))
    """
    Splits [([1 2,3], [4, 5, 6]),... ]
    """
    data_splits = [] # difficult to name and specify


    for train_idx, test_idx in splits: 
    
        # Further split training data into train and validation
        val_size_int = int(len(train_idx) * val_size) # val_size is used wrong here
        val_idx = train_idx[:val_size_int]
        train_idx = train_idx[val_size_int:]
  
        data_split =  {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
        }

        data_splits.append(data_split)
    return data_splits

def do_cross_fold_get_results(training_config, dataset):
    """
    Function that Trains Model on Different Folds of Dataset, returning the Best AUC for each fold. 


    Params:
        dataset (torch.Dataset??) - Pytorch Dataset for training ECG


    Returns: 
        best_aucs (list): list of aucs for each fold created
    """

    best_aucs = []  
    
    splits = create_data_splits(dataset, n_splits=5) # #copied
    
    for split in splits: 

        train_loader = training_config.generate_dataloader(dataset, split['train'])
        val_loader = training_config.generate_dataloader(dataset, split['val'])
        test_loader = training_config.generate_dataloader(dataset, split['test'])

        device = training_config.device
        model = training_config.generate_model().to(device)
        class_weights = training_config.class_weights.to(device)
        criterion = training_config.criterion.to(device) 
        optimizer = training_config.optimizer 


        best_auc = train_model(training_config, model, optimizer, criterion, train_loader, val_loader, device, save_model=False)
        best_aucs.append(best_auc)


    return best_aucs


def main():
    assert torch.cuda.is_available(), "Error: CUDA required to run trainer.py"

    training_config = TrainingConfig()

    dataset = training_config.get_dataset()

    # testing quickly to see if i have a dataloader working

    # implies that we have a single batch? what is the size of the batches? 

    # I need to have a new thing? why is it then that i only have 1 element in the batch?
#    splits = create_data_splits(dataset)
#    split = splits[0] # need to realize the structure of the dataset
#    dataloader = training_config.generate_dataloader(dataset, split['train'])
#    

    # TODO: Determine how to get the dataloader loading the right images from dataset?

    best_aucs = do_cross_fold_get_results(training_config, dataset)
    breakpoint()

if __name__ == '__main__':
    main()

