import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DuelingDeepQNetworkMlp(nn.Module):
    """
    Duelling Deep Q-Net Implementation with MLP

        Args:
            :arg [lr]: learning rate for the optimiser
            :arg [n_actions]: number of actions allowed by the env
            :arg [input_dims]: dimension for the conv net
    """

    def __init__(self, lr, n_actions, input_dims, fc1_dims=256, fc2_dims=128):
        super().__init__()

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        
        # Create state value and action advantage value streams
        self.V = nn.Linear(fc2_dims, 1)
        self.A = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.AdamW(self.parameters(), lr=lr, amsgrad=True)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        flat1 = F.relu(self.fc1(state))
        flat2 = F.relu(self.fc2(flat1))
        V = self.V(flat2)
        A = self.A(flat2)

        # Combine state value and advantage value to get final action values
        actions = V + A - A.mean(dim=1, keepdim=True)
        return actions
