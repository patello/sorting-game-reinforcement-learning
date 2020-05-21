from gamenet.agent import Agent
from gamenet.nn_runner import NNRunner
from gamenet.gameai import GameAI

if __name__ == "__main__":
    agent = Agent(base_net_file="test6",default_action_selection="Max")
    game_runner = GameAI()
    nnrunner = NNRunner(agent,game_runner)
    nnrunner.run_batch(100)