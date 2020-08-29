import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable

class IllegalMask(Exception):
    pass

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=100, state_dict=None):
        super(ActorCritic, self).__init__()

        self.hidden_size = hidden_size
        if(type(hidden_size)==int):
            actor_hidden_size = hidden_size
            critic_hidden_size = hidden_size
        elif(type(hidden_size)==list and len(hidden_size)==2):
            actor_hidden_size = hidden_size[0]
            critic_hidden_size = hidden_size[1]
        else:
            raise TypeError(hidden_size)

        self.num_actions = num_actions
        if type(actor_hidden_size)==int:
            self.actor_layers = nn.ModuleList([nn.Linear(num_inputs, actor_hidden_size), nn.Linear(actor_hidden_size, num_actions)])
        else:
            self.actor_layers = nn.ModuleList([nn.Linear(num_inputs, actor_hidden_size[0])] +[nn.Linear(actor_hidden_size[i], actor_hidden_size[i+1]) for i in range(len(actor_hidden_size)-1)] + [nn.Linear(actor_hidden_size[-1], num_actions)])        

        if type(critic_hidden_size)==int:
            self.critic_layers = nn.ModuleList([nn.Linear(num_inputs, critic_hidden_size), nn.Linear(critic_hidden_size, 1)])
        else:
            self.critic_layers = nn.ModuleList([nn.Linear(num_inputs, critic_hidden_size[0])] +[nn.Linear(critic_hidden_size[i], critic_hidden_size[i+1]) for i in range(len(critic_hidden_size)-1)] + [nn.Linear(critic_hidden_size[-1], 1)])
        
        if state_dict is not None:
            self.load_state_dict(state_dict)

    def forward_critic(self, state_tensor):
        value=state_tensor
        for i in range(len(self.critic_layers)-1):
            value = F.relu(self.critic_layers[i](value))
        return self.critic_layers[-1](value)

    def forward_actor(self, state_tensor, mask=None):
        #Not used anywhere, consider removing
        if mask is None:
            mask=torch.from_numpy(np.ones((1,self.num_actions),dtype='bool'))
        #Check that mask consist of at least one "True", otherwise forward will return NaN
        elif np.sum(mask.numpy()) == 0:
            raise IllegalMask
        policy=state_tensor
        for i in range(len(self.actor_layers)-1):
            policy = F.relu(self.actor_layers[i](policy))
        policy = self.actor_layers[-1](policy)
        policy[~mask]=float('-inf')
        policy_dist = F.softmax(policy, dim=1)
        #Also return the built-in log softmax since it is numerically stable
        log_policy_dist = F.log_softmax(policy, dim=1)
        return policy_dist, log_policy_dist