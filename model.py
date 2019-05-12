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

        self.input_layers = nn.Sequential(OrderedDict([
            ('i_conv1', nn.Conv2d(3, 16 , 3)),
            ('i_relu1', nn.ReLU()),
            ('i_pool1', nn.MaxPool2d(2, 2)),
            ('i_conv2', nn.Conv2d(16, 32, 8)),
            ('i_relu2', nn.ReLU()),
            ('i_pool2', nn.MaxPool2d(2, 2)),
            ('i_conv3', nn.Conv2d(32, 64, 8)),
            ('i_relu3', nn.ReLU()),
            ('i_pool3', nn.MaxPool2d(2, 2)),
            ('i_conv4', nn.Conv2d(64, 128, 8)),
            ('i_relu4', nn.ReLU()),
            ('i_pool4', nn.MaxPool2d(2, 2)),
            ('i_conv5', nn.Conv2d(128, 256, 3)),
            ('i_relu5', nn.ReLU()),
            ('i_pool5', nn.MaxPool2d(2, 2))
             ]))

        state_size = [512]
        self.hidden = []
        self.hidden.append(('fc_glue', nn.Linear(512, state_size[0])))
        self.hidden.append(('relu_glue', nn.ReLU()))
        for l in range(len(state_size) - 1):
            self.hidden.append(('fc' + str(l), nn.Linear(state_size[l], state_size[l + 1])))
            self.hidden.append(('relu' + str(l), nn.ReLU()))


        self.value_approximator_model = nn.Sequential(OrderedDict([
            ('logits', nn.Linear(state_size[-1], 1))]))


        self.advantage_approximator_model = nn.Sequential(OrderedDict([
            ('logits', nn.Linear(state_size[-1], self.output_size))]))

        self.feature_model = nn.Sequential(OrderedDict(self.hidden))

        print("self.model: {}".format(self.feature_model))

    def forward(self, state):
        """Build a network that maps state -> action values."""

        permitations = state.view(-1, state.size(3), state.size(1), state.size(2))
        feature_model_state = self.input_layers(permitations)
        dimension_squeeze = feature_model_state.view(len(feature_model_state), -1)
        state = self.feature_model(dimension_squeeze)
        advantege = self.advantage_approximator_model(state)
        value = self.value_approximator_model(state)

        return value + advantege - advantege.mean()
