import numpy as np  
import torch  
import torch.optim as optim

from torch.autograd import Variable
from gamenet.model import ActorCritic
from gamenet.nn_runner import NNRunner

class Agent():

    class AgentStatistics():
        def __init__(self):
            self.statisticsBuffer = {"reward" : np.empty(0), "actor_loss" : np.empty(0), "critic_loss" : np.empty(0), "entropy_loss": np.empty(0), "ac_loss" : np.empty(0)}
            self.statistics = {"reward" : np.empty(0), "actor_loss" : np.empty(0), "critic_loss" : np.empty(0), "entropy_loss": np.empty(0), "ac_loss" : np.empty(0)}
        def update(self,statistics):
            for stat in statistics:
                self.statisticsBuffer[stat]=np.append(self.statisticsBuffer[stat],statistics[stat])
        def get_stats(self):
            for stat in self.statistics:
                if len(self.statisticsBuffer[stat]) > 0:
                    self.statistics[stat] = np.append(self.statistics[stat],self.statisticsBuffer[stat].mean())
                    self.statisticsBuffer[stat] = np.empty(0)
            return self.statistics
                
    def __init__(self, model=None, learning_rate=3e-4, gamma=0.8, actor_loss_coeff=1, critic_loss_coeff=0.5, entropy_loss_coeff=0.01, default_action_selection = "Distribution"):
        self.agent_statistics = Agent.AgentStatistics()
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.actor_loss_coeff = actor_loss_coeff
        self.critic_loss_coeff = critic_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.default_action_selection = default_action_selection
        self.num_in = 20
        self.num_out = 16

        if model is None:
            self.ac_net = ActorCritic(self.num_in, self.num_out)
        else:
            self.ac_net = model
        self.ac_optimizer = optim.Adam(self.ac_net.parameters(), lr=learning_rate)
        
    def update(self, qvals, rewards, values, log_probs, entropy):
        values = torch.stack(values).squeeze(2)
        qvals = torch.FloatTensor(qvals)
        log_probs = torch.stack(log_probs)
        entropy = torch.stack(entropy)

        advantage = qvals - values
        
        actor_loss = (-log_probs * advantage.squeeze(1)).mean()
        critic_loss = advantage.pow(2).mean()
        #In the original implementation, they did not take the mean of the entropy. 
        #I'll use it here to normalize against number of episodes and batches
        #The factor of 0.1 is completely arbitrary
        entropy_loss = entropy.mean()
        ac_loss = self.actor_loss_coeff*actor_loss + self.critic_loss_coeff*critic_loss + self.entropy_loss_coeff*entropy_loss
        self.ac_optimizer.zero_grad()
        ac_loss.backward()
        self.ac_optimizer.step()
        statistics = {"reward" : np.mean(rewards), "actor_loss" : actor_loss.detach().numpy().squeeze(0), "critic_loss" : critic_loss.detach().numpy().squeeze(0), "entropy_loss":entropy_loss.detach().numpy().squeeze(0), "ac_loss" : ac_loss.detach().numpy().squeeze(0)}
        self.agent_statistics.update(statistics)

    def get_ac_output(self, state, valid_moves, action_selection=None):
        state = Variable(torch.from_numpy(np.array(state)).float().unsqueeze(0))
        value = self.ac_net.forward_critic(state)
        policy_dist, log_policy_dist = self.ac_net.forward_actor(state,valid_moves)
        if action_selection == "Distribution" or (action_selection == None and self.default_action_selection=="Distribution"):
            action = np.random.choice(self.num_out, p=policy_dist.detach().numpy().squeeze(0))
        elif action_selection == "Max" or (action_selection == None and self.default_action_selection=="Max"):
            action = np.argmax(policy_dist.detach().numpy().squeeze(0))
        return action, policy_dist, log_policy_dist, value

    def get_a_output(self, state, valid_moves, action_selection="Distribution"):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        policy_dist,_ = self.ac_net.forward_actor(state,valid_moves)
        if action_selection == "Distribution" or (action_selection == None and self.default_action_selection=="Distribution"):
            action = np.random.choice(self.num_out, p=policy_dist.detach().numpy().squeeze(0))
        elif action_selection == "Max" or (action_selection == None and self.default_action_selection=="Max"):
            action = np.argmax(policy_dist.detach().numpy().squeeze(0))
        return action

