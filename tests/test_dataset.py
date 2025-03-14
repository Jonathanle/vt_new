import pytest

from dataset import create_outcome, organize_patient_data
import os


@pytest.fixture(scope='module')
def labels_filepath():
    labels_filepath = os.environ.get('LABELS_FILEPATH')
    return labels_filepath


@pytest.fixture(scope='module')
def processed_data_dir():
    processed_data_dir = os.environ.get('PROCESSED_DATA_DIR')

    return processed_data_dir

def test_fixture_labels_filepath(labels_filepath): 

    assert isinstance(labels_filepath, str)
    assert len(labels_filepath) > 0

def test_create_organize_patient_data(processed_data_dir): 
    """
    show that the organize_patient_data creates a dict
    """

    patient_tensor_data = organize_patient_data(processed_data_dir)    
  
    
    assert isinstance(patient_tensor_data, dict)


def test_create_organize_patient_data_correct_shapes(processed_data_dir): 
    """
    show that the organize_patient_data creates a dict of right shapes

    (I can show later on that my process for verifying valid i manually checked via dict)
        

    Here i showed that cine_whole specifically does not have to be defined in any shape, 
    but that the other dimension is regularized to be 256 always 


    """

    patient_id = '11957' # -> these are specifically str ids not int ids
    
     
    patient_tensor_data = organize_patient_data(processed_data_dir)    
    patient_tensors = patient_tensor_data[patient_id] 
        
    for _, patient_tensors in patient_tensor_data.items():

        num_slices = patient_tensors['cine'].shape[0]

        for i in range(num_slices):

            # Heuristiically checks the 0th slice
            assert patient_tensors['cine'][i].shape == (128, 128)
            assert patient_tensors['lge'][i].shape == (128, 128)
            assert patient_tensors['raw'][i].shape == (128, 128)

            # Show first column dimension is flexible but that it should be greater than 128 for 
            #anything cropped to happen 
            assert patient_tensors['cine_whole'][i].shape[1] >= 128
            assert patient_tensors['cine_whole'][i].shape[0] >= 128





def test_create_outcome_returns_dict(labels_filepath): 
    """
    Future self - I showed you here that create_outcome helper returns a dict which helps to interface with the labels abstracting away any pd complexity.

    """
    outcome = create_outcome(labels_filepath)

    assert isinstance(outcome, dict)
def test_create_outcome_int_keys(labels_filepath): 
    """
    Future self - I showed you here that the outcome creates keys 
    """
    outcome = create_outcome(labels_filepath)

    for key in outcome.keys():
        assert isinstance(key, int)


def test_create_outcome_correct_label(labels_filepath): 
    """
    Test multiple sampled outcomes, especially testing on multiple copmosite outcomes

    In this case we are testing if the function creates composite outcomes that are correct according to sustainedvt | cardiacarrest | scd


    I heuristically assumed a couple of the composite outcomes and verfied it
    """

    # TODO: document the idea that the returned behavior of the xlsx is an integer to integer mapping
    ground_truth_values = {2188: 1, 2296: 0, 3282: 1, 3784: 1}
    outcome = create_outcome(labels_filepath)
    
    for patient_id, ground_truth_outcome in ground_truth_values.items():
        assert outcome[patient_id] == ground_truth_outcome
        


