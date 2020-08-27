from .model import ActorCritic
from .agent import Agent
from .nn_runner import NNRunner
from .sort_game import SortGame

import torch
import os

file_path = os.path.dirname(os.path.realpath(__file__))

def run_batch(model_path, batches=1000, state_fun="encoded", empty_pos_indicator = -41):
    model = torch.load(model_path)
    agent = Agent(model=model,default_action_selection="Max")
    game_runner = SortGame(empty_pos_indicator=empty_pos_indicator,state_fun=state_fun)
    nnrunner = NNRunner(agent,game_runner)
    nnrunner.run_batch(batches)

def train(model=None,model_path=None,result_path=None, state_fun="encoded", empty_pos_indicator = -41, hidden_size=200, batch_size=10,batches=10000,learning_rate=3e-5, gamma=0.9, entropy_loss_coeff=0.01):
    if state_fun=="encoded":
        input_size=816
    elif state_fun=="int":
        input_size=20
    if model!=None:
        model=torch.load(file_path+model_path)
    else:
        model = ActorCritic(input_size,16,hidden_size=hidden_size)
    agent = Agent(model=model,learning_rate=learning_rate, gamma=gamma, entropy_loss_coeff=entropy_loss_coeff)
    game_runner = SortGame(empty_pos_indicator=empty_pos_indicator,state_fun=state_fun)
    nnrunner = NNRunner(agent,game_runner)
    #model_path argument tells the nnrunner where to save the trained model, relative to current directory (saved every 1000th batch)
    #result_path argument tells nnrunner where to save a csv file with training data over time, otherwise outputs to terminal
    nnrunner.train(batch_size=batch_size,batches=batches,model_path=model_path,result_path=result_path)
