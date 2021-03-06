import random
import copy
import numpy as np
import torch  
import csv
import os
import itertools
import math

random.seed()

# Class till will help the neural network to run properly
class NNRunner():    
    def __init__(self,agent,game_runner):
        self.agent=agent
        self.game_runner=game_runner
    def run_episode(self):
        rewards = []
        values = []
        log_probs = []
        entropy_terms = []
        self.game_runner.reset()
        state = np.array(self.game_runner.get_state())
        #Run until step returns done=True
        done = False
        while not done:
            valid_moves = torch.from_numpy(np.array(self.game_runner.get_valid_moves()).reshape(1,16))
            action, policy_dist, log_policy_dist, value = self.agent.get_ac_output(state,valid_moves)
            reward, done = self.game_runner.step(action)  
            new_state = self.game_runner.get_state()

            log_prob = log_policy_dist.squeeze(0)[action]
            
            #When calculating entropy, we need to only look at the valid policy distribution. Otherwise, entropy is infinite
            #Masking seems to be needed even when the built in log_softmax is used.
            valid_log_policy_dist = log_policy_dist.masked_select(valid_moves)
            
            #Using the mean() funciton, Should give same result as the original code:
            #-torch.sum(policy_dist.masked_select(valid_moves).mean() * torch.log(policy_dist.masked_select(valid_moves)))
            entropy = -valid_log_policy_dist.mean()

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_terms.append(entropy)
            state = new_state
        statistics = {"reward":sum(rewards)}
        self.agent.agent_statistics.update(statistics)
        return rewards, values, log_probs, entropy_terms
    def run_batch(self, episodes):
        for episode in range(episodes):
            self.run_episode()
        mean_reward = self.agent.agent_statistics.get_stats()["reward"][-1]
        print("Mean reward: " + str(mean_reward))
        return mean_reward
    def train(self, result_path=None, model_path=None, batch_size=10, batches=1000, break_cond = None):
        if result_path is not None:
            if os.path.exists(result_path):
                raise FileExistsError(result_path)
            else:
                with open(result_path, mode="w") as csv_file:
                            result_file = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            result_file.writerow(["batch"]+list(self.agent.agent_statistics.statistics.keys()))
        if model_path is not None:
            if os.path.exists(model_path):
                raise FileExistsError(model_path)
            else:
                torch.save(self.agent.ac_net,model_path)
        if batches > 0:
            batch_range=range(batches)
        else:
        #If batches are -1, then loop infinitly
            batch_range=itertools.count()
        if break_cond is not None:
            prev_reward = 0
        for batch in batch_range:
            rewards = []
            values = []
            log_probs = []
            qvals = np.array([])
            entropy_term = []
            for episode in range(batch_size):
                episode_rewards, episode_values, episode_log_probs, episode_entropy_term = self.run_episode()
                rewards.append(np.sum(episode_rewards))
                values += episode_values
                log_probs += episode_log_probs
                entropy_term += episode_entropy_term
                episode_qvals = np.zeros(len(episode_rewards))

                qval = 0
                for t in reversed(range(len(episode_rewards))):
                    qval = episode_rewards[t] + self.agent.gamma * qval
                    episode_qvals[t] = qval
                qvals = np.concatenate((qvals,episode_qvals))

            self.agent.update(np.reshape(qvals,(-1,1)), rewards, values, log_probs, entropy_term)
            if (batch+1) % min(1000,math.ceil(batches/1000)) == 0:    
                if result_path is not None:                
                    with open(result_path, mode="a+") as csv_file:
                        result_file = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        result_file.writerow(np.concatenate([[batch+1],[stat_value[-1] for stat_value in self.agent.agent_statistics.get_stats().values()],]))
                else:
                    print(str(batch+1)+": " + str(self.agent.agent_statistics.get_stats()["reward"][-1]))
                if model_path is not None:
                    model_dict = {
                        "state_dict" : self.agent.ac_net.state_dict(),
                        "state_fun" : self.game_runner.state_fun,
                        "empty_pos_indicator" : self.game_runner.empty_pos_indicator,
                        "hidden_size" : self.agent.ac_net.hidden_size,
                        "gamma" : self.agent.gamma,
                        "entropy_loss_coeff" : self.agent.entropy_loss_coeff,
                    }
                    torch.save(model_dict,model_path)
        return (batches,self.agent.agent_statistics.get_stats()["reward"][-1])