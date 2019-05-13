import torch
import torch.nn as nn
from collections import OrderedDict


class DuelQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)


        self.input_size = state_size
        self.output_size = action_size

        self.layer_sizes = [self.input_size, self.input_size * self.input_size, self.input_size * self.input_size * self.output_size, self.output_size * self.input_size, self.output_size]
        self.layers = []
        for l in range(len(self.layer_sizes) - 1):
            self.layers.append(('fc' + str(l), nn.Linear(self.layer_sizes[l], self.layer_sizes[l + 1])))
            self.layers.append(('relu' + str(l), nn.ReLU()))

        self.layers.append(('Dropout' + str(l), nn.Dropout(0.3)))


        self.value_approximator_model = nn.Sequential(OrderedDict([
            ('logits', nn.Linear(self.layer_sizes[-1], 1))]))


        self.advantage_approximator_model = nn.Sequential(OrderedDict([
            ('logits', nn.Linear(self.layer_sizes[-1], self.output_size))]))

        self.feature_model = nn.Sequential(OrderedDict(self.layers))

        print("self.model: {}".format(self.feature_model))

    def forward(self, state):
        """Build a network that maps state -> action values."""
        output = self.feature_model(state)
        advantege = self.advantage_approximator_model(output)
        value = self.value_approximator_model(output)

        return value + advantege - advantege.mean()
