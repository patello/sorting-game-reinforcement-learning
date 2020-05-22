from gamenet.model import ActorCritic
from gamenet.agent import Agent
from gamenet.nn_runner import NNRunner
from gamenet.gameai import GameAI

model = ActorCritic(20,16,hidden_size=[[80,80],100])
agent = Agent(model=model,learning_rate=3e-5, gamma=0.9, entropy_loss_coeff=0.01)
game_runner = GameAI(empty_pos_indicator=-41)
nnrunner = NNRunner(agent,game_runner)
#model_path argument tells the nnrunner where to save the trained model, relative to current directory (saved every 1000th batch)
#result_path argument tells nnrunner where to save a csv file with training data over time, otherwise outputs to terminal
nnrunner.train(batch_size=1,batches=2000,model_path="./results/test.mx",result_path="./results/test.csv")
