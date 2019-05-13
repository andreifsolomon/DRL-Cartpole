import numpy as np
import random
from collections import namedtuple, deque

from model import  DuelQNetwork

# from model import DuelQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
from segment_tree import *

from prioritized_replay_buffer import PrioritizedReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, double_dqn=False):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.double_dqn = double_dqn

        self.state_size = state_size[0]
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = DuelQNetwork(self.state_size, self.action_size, seed).to(device)
        self.qnetwork_target = DuelQNetwork(self.state_size, self.action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        # self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, alpha=1)
        self.memory = PrioritizedReplayBuffer(100000)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        # self.memory.add(state, action, reward, next_state, done)
        experience = state, action, reward, next_state, done
        self.memory.store(experience)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            # if len(self.memory) > BATCH_SIZE:
                # experiences = self.memory.sample()
            tree_idx, batch, ISWeights_mb = self.memory.sample(BATCH_SIZE)
            self.learn(tree_idx, list(zip(*batch)), GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if action_values is not None and random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, indexes, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        states = torch.tensor(states).float().to(device)
        actions = torch.tensor(actions).long().to(device)
        rewards = torch.tensor(rewards).float().to(device)
        next_states = torch.tensor(next_states).float().to(device)
        dones = torch.tensor(dones).float().to(device)

        if self.double_dqn:
            _, best_actions = self.qnetwork_local(next_states).detach().max(dim=1)
            Q_targets_next = self.qnetwork_target(states).gather(1, best_actions.view(-1,1))

            # Compute Q targets for current states
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(states).gather(1, actions)

            for i in range(BATCH_SIZE):
                if dones[i].item() == 1.0:
                    Q_targets[i] = rewards[i]

            delta = torch.clamp(Q_targets - Q_expected, -1., 1.)
            loss = torch.sum(torch.pow(delta, 2))

            self.optimizer.zero_grad()
            loss.backward()

        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            # Compute Q targets for current states
            # Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            Q_targets = rewards.view(-1, 1) + (gamma * Q_targets_next * (1 - dones.view(-1, 1)))

            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(states).gather(1, actions.view(-1,1))

            # Compute loss
            # loss = F.mse_loss(Q_expected, Q_targets)
            delta = torch.clamp(Q_targets - Q_expected, -1., 1.)
            loss = torch.pow(delta, 2)

            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()

        loss_data = loss.detach().squeeze(1).data.numpy()
        for i in range(len(loss_data)):
            update_index = indexes[i]
            priority = loss_data[i]
            self.memory.update(update_index, priority)


        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# class PrioritizedReplayBuffer(ReplayBuffer):
#     def __init__(self, action_size, buffer_size, batch_size, seed, alpha):
#         """Create Prioritized Replay buffer.
#         Parameters
#         ----------
#         size: int
#             Max number of transitions to store in the buffer. When the buffer
#             overflows the old memories are dropped.
#         alpha: float
#             how much prioritization is used
#             (0 - no prioritization, 1 - full prioritization)
#         See Also
#         --------
#         ReplayBuffer.__init__
#         """
#         super(PrioritizedReplayBuffer, self).__init__(action_size, buffer_size, batch_size, seed)
#         assert alpha >= 0
#         self._alpha = alpha
#
#         it_capacity = 1
#         while it_capacity < buffer_size:
#             it_capacity *= 2
#
#         self._it_sum = SumSegmentTree(it_capacity)
#         self._it_min = MinSegmentTree(it_capacity)
#         self._max_priority = 1.0
#
#     def add(self, *args, **kwargs):
#         """See ReplayBuffer.store_effect"""
#         idx = self._next_idx
#         super().add(*args, **kwargs)
#         self._it_sum[idx] = self._max_priority ** self._alpha
#         self._it_min[idx] = self._max_priority ** self._alpha
#
#     def _sample_proportional(self, batch_size):
#         res = []
#         p_total = self._it_sum.sum(0, len(self._storage) - 1)
#         every_range_len = p_total / batch_size
#         for i in range(batch_size):
#             mass = random.random() * every_range_len + i * every_range_len
#             idx = self._it_sum.find_prefixsum_idx(mass)
#             res.append(idx)
#         return res
#
#     def sample(self, batch_size, beta):
#         """Sample a batch of experiences.
#         compared to ReplayBuffer.sample
#         it also returns importance weights and idxes
#         of sampled experiences.
#         Parameters
#         ----------
#         batch_size: int
#             How many transitions to sample.
#         beta: float
#             To what degree to use importance weights
#             (0 - no corrections, 1 - full correction)
#         Returns
#         -------
#         obs_batch: np.array
#             batch of observations
#         act_batch: np.array
#             batch of actions executed given obs_batch
#         rew_batch: np.array
#             rewards received as results of executing act_batch
#         next_obs_batch: np.array
#             next set of observations seen after executing act_batch
#         done_mask: np.array
#             done_mask[i] = 1 if executing act_batch[i] resulted in
#             the end of an episode and 0 otherwise.
#         weights: np.array
#             Array of shape (batch_size,) and dtype np.float32
#             denoting importance weight of each sampled transition
#         idxes: np.array
#             Array of shape (batch_size,) and dtype np.int32
#             idexes in buffer of sampled experiences
#         """
#         assert beta > 0
#
#         idxes = self._sample_proportional(batch_size)
#
#         weights = []
#         p_min = self._it_min.min() / self._it_sum.sum()
#         max_weight = (p_min * len(self._storage)) ** (-beta)
#
#         for idx in idxes:
#             p_sample = self._it_sum[idx] / self._it_sum.sum()
#             weight = (p_sample * len(self._storage)) ** (-beta)
#             weights.append(weight / max_weight)
#         weights = np.array(weights)
#         encoded_sample = self._encode_sample(idxes)
#         return tuple(list(encoded_sample) + [weights, idxes])
#
#     def update_priorities(self, idxes, priorities):
#         """Update priorities of sampled transitions.
#         sets priority of transition at index idxes[i] in buffer
#         to priorities[i].
#         Parameters
#         ----------
#         idxes: [int]
#             List of idxes of sampled transitions
#         priorities: [float]
#             List of updated priorities corresponding to
#             transitions at the sampled idxes denoted by
#             variable `idxes`.
#         """
#         assert len(idxes) == len(priorities)
#         for idx, priority in zip(idxes, priorities):
#             assert priority > 0
#             assert 0 <= idx < len(self._storage)
#             self._it_sum[idx] = priority ** self._alpha
#             self._it_min[idx] = priority ** self._alpha
#
#             self._max_priority = max(self._max_priority, priority)