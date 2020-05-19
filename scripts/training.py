from gamenet.agent import Agent
from gamenet.nn_runner import NNRunner
from gamenet.gameai import GameAI


agent = Agent(base_net_file=None,learning_rate=3e-4, gamma=0.8)
game_runner = GameAI()
nnrunner = NNRunner(agent,game_runner)
nnrunner.train(batch_size=20,batches=100000,net_name="test4")
