import torch
import numpy as np

from flask import Flask, Markup, render_template, jsonify, request

import os,sys

sys.path.append(os.path.dirname(__file__))
from sort_game_net.agent import Agent
from sort_game_net.nn_runner import NNRunner
from sort_game_net.sort_game import SortGame
from sort_game_net.model import ActorCritic

app = Flask(__name__)

model_dict = torch.load(os.path.join(os.path.dirname(__file__), "pre_trained", "binary_encoded.pth"))
if model_dict["state_fun"]=="encoded":
   input_size=816
elif model_dict["state_fun"]=="int":
    input_size=20
model = ActorCritic(input_size,16,model_dict["hidden_size"],model_dict["state_dict"])
agent = Agent(model=model,default_action_selection="Max")
game_runner = SortGame(empty_pos_indicator=model_dict["empty_pos_indicator"],state_fun=model_dict["state_fun"])
game_runner.reset()

@app.route('/', methods=['GET', 'POST'])
def index():
    data = request.get_json()
    if data:
        game_runner.board = data["board"]
        game_runner.currBricks[0] = data["selected_brick"]
        game_runner.currBricks[1:len(data["other_bricks"])+1] = data["other_bricks"]
        game_runner.currBricks[len(data["other_bricks"])+1:4]=[0]*len(data["other_bricks"])
        state = np.array(game_runner.get_state())
        action, policy_dist, log_policy_dist, value = agent.get_ac_output(state, torch.from_numpy(np.array(game_runner.get_valid_moves()).reshape(1,16)))
        return jsonify({"action":int(action),"policy":policy_dist.tolist()[0], "value":value.tolist()[0][0]})