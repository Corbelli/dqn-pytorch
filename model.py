import torch
import torch.nn as nn
import torch.nn.functional as F

class DuellingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuellingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fp1 = nn.Linear(state_size, 30)
        # self.fp2 = nn.Linear(256, 256)
        # self.fp3 = nn.Linear(256, 256)
        # self.fp4 = nn.Linear(256, 256)
        # self.fp5 = nn.Linear(256, 256)
        # self.head_value = nn.Linear(64, 1)
        self.head_advantages = nn.Linear(30, action_size)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fp1(state))
        # x = F.relu(self.fp2(x))
        # x = F.relu(self.fp3(x))
        # x = F.relu(self.fp4(x))
        # x = F.relu(self.fp5(x))
        # state_value = self.head_value(x)
        state_advantages = self.head_advantages(x)
        return state_advantages
        # return state_value + (state_advantages - state_advantages.mean())
