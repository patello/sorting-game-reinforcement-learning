from gamenet.agent import Agent
from gamenet.nn_runner import NNRunner
from gamenet.gameai import GameAI

import torch

if __name__ == "__main__":
    model = torch.load("./examples/examplemodel.mx")
    agent = Agent(model=model,default_action_selection="Max")
    #empty_pos_indicator needs to be parametrized to the same setting as the model was trained with
    game_runner = GameAI(empty_pos_indicator=-41)
    nnrunner = NNRunner(agent,game_runner)
    nnrunner.run_batch(1000)