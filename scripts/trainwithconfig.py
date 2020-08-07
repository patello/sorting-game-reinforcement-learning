from gamenet.model import ActorCritic
from gamenet.agent import Agent
from gamenet.nn_runner import NNRunner
from gamenet.gameai import GameAI
from gamenet.configmanager import ConfigManager
import sys

cfg_file = sys.argv[1]

cfg = ConfigManager(cfg_file)

model = ActorCritic(20,16,hidden_size=cfg.hidden_size)
agent = Agent(model=model,learning_rate=cfg.learning_rate, gamma=cfg.gamma, entropy_loss_coeff=cfg.entropy_loss_coeff)
game_runner = GameAI(empty_pos_indicator=cfg.empty_pos_indicator)
nnrunner = NNRunner(agent,game_runner)
nnrunner.train(batch_size=cfg.batch_size,batches=-1,break_cond=1,model_path="./results/"+cfg.name+".mx",result_path="./results/"+cfg.name+".csv")
