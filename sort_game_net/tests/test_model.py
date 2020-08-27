import pytest
import numpy as np
import torch

from ..model import ActorCritic

model = ActorCritic(2,3,4)

@pytest.mark.parametrize("hidden_size", [10,[10,20],[[10,20],[20,10]]])
def test_model_init(hidden_size):
    model = ActorCritic(20,16,hidden_size)
    
def test_forward_critic():
    state = torch.autograd.Variable(torch.from_numpy(np.array([1,1])).float().unsqueeze(0))
    res = model.forward_critic(state)
    assert res.size() == torch.Size([1,1])

def test_forward_actor():
    state = torch.autograd.Variable(torch.from_numpy(np.array([1,1])).float().unsqueeze(0))
    mask = torch.from_numpy(np.array([True,True,True]).reshape(1,3))
    res = model.forward_actor(state,mask)
    assert res[0].size() == torch.Size([1,3])
    assert res[1].size() == torch.Size([1,3])