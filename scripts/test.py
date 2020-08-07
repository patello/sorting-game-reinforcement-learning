from gamenet.agent import Agent
from gamenet.nn_runner import NNRunner
from gamenet.gameai import GameAI

import torch
import numpy as np

if __name__ == "__main__":
    model = torch.load("./examples/run86.mx")
    agent = Agent(model=model,default_action_selection="Max")
    board = [-41]*16+[40,27,4,31]
    action,policy_dist,_,_=agent.get_ac_output(board,torch.from_numpy(np.array([board[i] == -41 for i in range(16)]).reshape(1,16)))
    print(action)