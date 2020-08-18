import torch
import numpy as np

from flask import Flask, Markup, render_template, jsonify, request

import sys
sys.path.append("/root/projects/gameaiclass/")
from gamenet.agent import Agent
from gamenet.nn_runner import NNRunner
from gamenet.gameai import GameAI

app = Flask(__name__)

model = torch.load("/root/projects/gameaiclass/results/encodingtest2.mx")
agent = Agent(model=model,default_action_selection="Max")
game_runner = GameAI(empty_pos_indicator=-0,state_fun="encoded")
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