import torch
import numpy as np
import os

from ..agent import Agent

file_path = os.path.dirname(os.path.realpath(__file__))
agent = Agent()

def test_agent_init():
    #Test normal initialization
    agent = Agent()

def test_agent_update():
    assert(True)
    #Add test, maybe by reading dict?

def test_agent_get_ac_output():
    state=np.array([0]*20)
    valid_moves = torch.from_numpy(np.array([True]*16).reshape(1,16))
    res = agent.get_ac_output(state, valid_moves)
    assert(type(res[0])==int)
    assert(res[1].size()==torch.Size([1,16]))
    assert(res[2].size()==torch.Size([1,16]))
    assert(res[3].size()==torch.Size([1,1]))

def test_agent_get_a_output():
    state=np.array([0]*20)
    valid_moves = torch.from_numpy(np.array([True]*16).reshape(1,16))
    res = agent.get_a_output(state, valid_moves)
    assert(type(res)==int)
    