from gamenet.agent import Agent
from gamenet.nn_runner import NNRunner
from gamenet.gameai import GameAI

import torch
import numpy

act_sum=torch.zeros(1,400)

def printnorm(self, input, output):
    global act_sum
    act_sum+=torch.nn.functional.relu(output)

if __name__ == "__main__":
    model = torch.load("./results/encodingtest2.mx")
    model.actor_layers[0].register_forward_hook(printnorm)
    agent = Agent(model=model,default_action_selection="Max")
    #empty_pos_indicator needs to be parametrized to the same setting as the model was trained with
    game_runner = GameAI(empty_pos_indicator=-41,state_fun="encoded")
    nnrunner = NNRunner(agent,game_runner)
    nnrunner.run_batch(1000)
    print(numpy.count_nonzero(act_sum.detach().numpy()>0))