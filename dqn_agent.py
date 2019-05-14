import numpy as np
import random
from collections import namedtuple, deque

from model import  DuelQNetwork

# from model import DuelQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
from segment_tree import *

# from prioritized_replay_buffer import PrioritizedReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512  # minibatch size
GAMMA = 0.99  # discount factor
ALPHA = 0.5 # prioritization level (ALPHA=0 is uniform sampling so no prioritization)
TAU = 1e-3  # for soft update of target parameters
LR = 5e-3  # learning rate
UPDATE_EVERY = 10  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        self.state_size = state_size[0]
        self.action_size = action_size
        self.seed = random.seed(seed)

        # DuelQNetwork
        self.qnetwork_local = DuelQNetwork(self.state_size, self.action_size, seed).to(device)
        self.qnetwork_target = DuelQNetwork(self.state_size, self.action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Prioritized Experienced Replay memory
        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, beta=1.):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(ALPHA, beta)
                # Is line below required? Don't think so looks like no-op ...
                # action_values = self.qnetwork_local(experiences[0])
                self.learn(experiences, GAMMA)

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

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights, indices = experiences

        # Get max action from local model
        local_max_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = torch.gather(self.qnetwork_target(next_states).detach(), 1, local_max_actions)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        # loss = F.mse_loss(Q_expected, Q_targets)
        loss = (Q_expected - Q_targets).pow(2) * weights
        adjusted_loss = loss + 1e-5
        loss = loss.mean()

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities based on td error
        self.memory.update_priorities(indices.squeeze().to(device).data.numpy(), adjusted_loss.squeeze().to(device).data.numpy())

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


class PrioritizedReplayBuffer:
    """Naive Prioritized Experience Replay buffer to store experience tuples."""

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
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done", "priority"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # By default set max priority level
        max_priority = max([m.priority for m in self.memory]) if self.memory else 1.0
        e = self.experience(state, action, reward, next_state, done, max_priority)
        self.memory.append(e)

    def sample(self, alpha, beta):
        """Randomly sample a batch of experiences from memory."""

        # Probabilities associated with each entry in memory
        priorities = np.array([sample.priority for sample in self.memory])
        probs = priorities ** alpha
        probs /= probs.sum()

        # Get indices
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False, p=probs)

        # Associated experiences
        experiences = [self.memory[idx] for idx in indices]

        # Importance sampling weights
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        weights = torch.from_numpy(np.vstack(weights)).float().to(device)
        indices = torch.from_numpy(np.vstack(indices)).long().to(device)
        return (states, actions, rewards, next_states, dones, weights, indices)

    def update_priorities(self, indices, priorities):
        for i, idx in enumerate(indices):
            # A tuple is immutable so need to use "_replace" method to update it - might replace the named tuple by a dict
            self.memory[idx] = self.memory[idx]._replace(priority=priorities[i])

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


#from: https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
#credits to: Jaromir Janisch
"""
The Memory that is used for Prioritized Experience Replay
"""

from sum_tree import SumTree
class Replay_Memory:
    def __init__(self):
        global MEMORY_LEN
        self.tree = SumTree(MEMORY_LEN)

    def add(self, error, sample):
        global MEMORY_BIAS, MEMORY_POW
        priority = (error + MEMORY_BIAS) ** MEMORY_POW
        self.tree.add(priority, sample)

    def sample(self):
        """
         Get a sample batch of the replay memory
        Returns:
         batch: a batch with one sample from each segment of the memory
        """
        global BATCH_SIZE
        batch = []
        #we want one representative of all distribution-segments in the batch
        #e.g BATCH_SIZE=2: batch contains one sample from [min,median]
        #and from [median,max]
        segment = self.tree.total() / BATCH_SIZE
        for i in range(BATCH_SIZE):
            minimum = segment * i
            maximum = segment * (i+1)
            s = random.uniform(minimum, maximum)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))
        return batch

    def update(self, idx, error):
        """
         Updates one entry in the replay memory
        Args:
         idx: the position of the outdated transition in the memory
         error: the newly calculated error
        """
        priority = (error + MEMORY_BIAS) ** MEMORY_POW
        self.tree.update(idx, priority)