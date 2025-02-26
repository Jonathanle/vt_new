import pytest

import torch



test_torch_cuda_is_available(): 
    assert torch.cuda.is_available() 
