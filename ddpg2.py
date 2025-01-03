import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size ,init_w = 3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w,init_w)
        self.linear3.bias.data.uniform_(-init_w,init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w = 3e-3):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        # nn.init.uniform_(self.linear3.weight,(-init_w,init_w))
        self.linear3.bias.data.uniform_(-init_w, init_w)
        

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0,0]



# 叠加噪声
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta = 0.15, max_sigma = 0.3, min_sigma = 0.3, decay_period = 100000):#decay_period要根据迭代次数合理设置
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) *self.mu

    def evolve_state(self):   
        x = self.state
        dx = self.theta* (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):   
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


# 建立回访缓存器类存储行为策略产生的数据
class ReplayBuffer:
    def __init__(self, capacity):  
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# 标准化动作类
class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low_bound = self.action_space.low     
        upper_bound = self.action_space.high   

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        #将经过tanh输出的值重新映射回环境的真实值内lw
        action = np.clip(action, low_bound, upper_bound)

        return action

    def reverse_action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)

        return action


class DDPG(object):
    def __init__(self, action_dim, state_dim, hidden_dim): 
        super(DDPG, self).__init__()
        self.action_dim, self.state_dim, self.hidden_dim = action_dim, state_dim, hidden_dim
        self.batch_size = 128
        self.gamma = 0.99
        self.min_value = -np.inf    # -inf
        self.max_value = np.inf     # inf
        self.soft_tau = 1e-2
        self.replay_buffer_size = 5000
        self.value_lr = 1e-3
        self.policy_lr = 1e-4

        # critic网络，actor网络，
        self.value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device) # (1,3,256)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        # critic目标网络，actor目标网络，
        self.target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        # 为预测网络选择Adam梯度下降算法
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)

        # critic网络的损失函数
        self.value_criterion = nn.MSELoss()   

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)   

    def ddpg_update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)  

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()

        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, self.min_value, self.max_value)

        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # 根据滑动平均更新目标网络参数
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )


def plot(frame_idx, rewards):
    plt.figure(figsize=(5,5)) 
    # plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    #plt.show()
    plt.savefig('./%s.png' % frame_idx)


def main():
    env = gym.make("Pendulum-v1", render_mode='human')
    env = NormalizedActions(env)  

    ou_noise = OUNoise(env.action_space)  # 叠加噪声

    state_dim = env.observation_space.shape[0]    
    action_dim = env.action_space.shape[0]        
    hidden_dim = 256

    ddpg = DDPG(action_dim, state_dim, hidden_dim)   

    max_frames = 100
    max_steps = 500
    frame_idx = 0
    rewards = []
    batch_size = 128

    while frame_idx < max_frames:
        state = env.reset()
        state = state[0]
        ou_noise.reset()
        episode_reward = 0

        for step in range(max_steps):
            env.render()
            action = ddpg.policy_net.get_action(state)   # 根据当前的actor网络->策略选择动作
            action = ou_noise.get_action(action, step)   # 对动作添加噪音
            next_state, reward, done, info, _ = env.step(action)   # 执行动作，观测新状态，回报

            ddpg.replay_buffer.push(state, action, reward, next_state, done)  # 交互数据存储到经验缓存器
            # 从经验缓存器R中随机采样128个mini_batch数据(s_i,a_i,r_i,s_(i+1))
            if len(ddpg.replay_buffer) > batch_size:
                ddpg.ddpg_update()   # 更新网络

            state = next_state
            episode_reward += reward
            frame_idx += 1


            if frame_idx % max(1000, max_steps + 1) == 0:
                plot(frame_idx, rewards)

            if done:
                break

        rewards.append(episode_reward)
    plot(frame_idx, rewards)
    env.close()


if __name__ == '__main__':
    main()



