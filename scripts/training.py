from gamenet.model import ActorCritic
from gamenet.agent import Agent
from gamenet.nn_runner import NNRunner
from gamenet.gameai import GameAI

model = ActorCritic(20,16,hidden_size=[[80,80],100])
agent = Agent(model=model,learning_rate=3e-5, gamma=0.9, entropy_loss_coeff=0.01)
game_runner = GameAI(empty_pos_indicator=-41)
nnrunner = NNRunner(agent,game_runner)
nnrunner.train(batch_size=20,batches=1000000,net_name="test8")
