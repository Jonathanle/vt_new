import pytest
import torch

def test_torch_cuda_is_available():
    assert torch.cuda.is_available() 
