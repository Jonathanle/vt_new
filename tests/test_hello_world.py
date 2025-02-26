import pytest


from hello_world import add 
# how to change and get the original directory 


def test_adding(): 
    sum = add(2,3)
    assert sum == 5
