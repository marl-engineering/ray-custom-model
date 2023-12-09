import torch as T
import numpy as np

from src.buffer.generic import ReplayBuffer
from src.net.ddqn import DuelingDeepQNetworkMlp


class DuellingDDQNAgent:
    def __init__(self, config):
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.lr = config.lr
        self.n_actions = config.n_actions
        self.input_dims = config.input_dims
        self.batch_size = config.batch_size
        self.eps_min = config.eps_min
        self.eps_dec = config.eps_dec
        self.mem_size = config.mem_size
        self.replace_target_cnt = config.replace
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(self.mem_size, self.input_dims, self.n_actions)

        self.q_eval = DuelingDeepQNetworkMlp(self.lr, self.n_actions,
                                    input_dims=self.input_dims)

        self.q_next = DuelingDeepQNetworkMlp(self.lr, self.n_actions,
                                    input_dims=self.input_dims)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = \
                            self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done.astype(np.bool_)).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones


    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]

        with T.no_grad():
            q_next = self.q_next.forward(states_).max(dim=1)[0]
            q_next.masked_fill_(dones, 0.0)
            q_target = (rewards + self.gamma * q_next)
        
        loss = self.custom_mse_loss(q_pred, q_target).to(self.q_eval.device)
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.q_eval.parameters(), 10.0)
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
    
    def custom_mse_loss(self, inpt, target):
        return (inpt - target).pow(2).mean()
