#小车滑动的demo，

# https://blog.csdn.net/qq_43459731/article/details/135181631

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import time
 
# GPU设置
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
 
# 超参数
BATCH_SIZE = 60                                 # 样本数量 (每次从记忆库中抽取的样本数量喂入q网络)
LR = 0.01                                       # 学习率
EPSILON = 0.9                                   # greedy policy  也叫探索率，有一定概率探索新路径
GAMMA = 0.9                                     # reward discount
TARGET_REPLACE_ITER = 100                       # 目标网络更新频率 (每100轮后就将评估网络的参数复制给目标网络)
MEMORY_CAPACITY = 500                           # 记忆库容量
 
# 和环境相关的参数
env = gym.make("CartPole-v1",render_mode="human").unwrapped   # 使用gym库中的环境：CartPole，且打开封装(若想了解该环境，请自行百度)
"""
这里是调用了OPENAI封装好的一个库，对于平衡车这个demo里面已经初始化了环境

gym.make 用于创建指定"CartPole-v1"环境
render_mode="human" 表示在创建环境时使用人类可视化，即在图形窗口中渲染环境
.unwrapped 操作是为了获取环境的未封装版本 
未封装的环境是原始的、没有经过额外修改的环境对象
用户直接与原始环境进行交互，可以更直接地使用原始环境的状态、动作等信息
没有附加的功能或修改，是环境的基本形式
"""
 
N_state = env.observation_space.shape[0]      # 环境的特征数和维度
N_action = env.action_space.n  # 环境的动作数量

# 估计动作值函数

#其实就是一个全连接层，隐藏层的维度是50，激活函数是ReLU

class Net(nn.Module):
    def __init__(self):
 
        super(Net,self).__init__()
        self.fc1 = nn.Linear(N_state,50) # 定义全连接层 (是输入层到隐藏层的线性层，输入大小为 N_state（状态空间的维度），输出大小为 50)
        self.fc1.weight.data.normal_(0,0.1) # 对权重进行了正态分布初始化，均值为 0，标准差为 0.1
        self.out = nn.Linear(50,N_action) # 定义全连接层 (是隐藏层到输出层的线性层，输入大小为 50，输出大小为 N_action（动作空间的大小))
        self.out.weight.data.normal_(0,0.1) # # 对权重进行了正态分布初始化，均值为 0，标准差为 0.1
 
    def forward(self,x):
        x = F.relu(self.fc1(x)) # 通过隐藏层 self.fc1 使用 ReLU 激活函数进行非线性转换
        action_value = self.out(x) # 将非线性转换后的结果传递给输出层 self.out，得到动作值（Q 值）
        return action_value # 返回动作值作为网络的输出
 
 
# 定义DQN类(定义Q网络以及一个固定的Q网络)
class DQN(object):
    def __init__(self):

        #在DQN中会有两个网络，分别是评估网络和目标网络

        # 创建实时网络和目标网络
        # self.eval_net 用于实时更新 Q 值
        # self.target_net 是一个固定的目标网络，用于稳定训练
        self.eval_net,self.target_net = Net().to(device),Net().to(device)

        self.learn_step_counter = 0  # 学习步数记录 (控制目标网络的更新频率)
        self.memory_counter = 0      # 记忆量计数 (判断是否需要进行经验回放)

        self.memory = np.zeros((MEMORY_CAPACITY,N_state*2+2)) # 存储空间初始化，每一组的数据为(s_t,a_t,r_t,s_{t+1})
        self.optimazer = torch.optim.Adam(self.eval_net.parameters(),lr=LR) # 更新评估网络的参数，学习率为 LR
        self.loss_func = nn.MSELoss()     # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
        self.loss_func = self.loss_func.to(device) # 衡量评估网络输出和目标 Q 值之间的误差
 
    """ 
    def choose_action() 在给定状态 x 的情况下选择动作
    """
    def choose_action(self,x):
        x = torch.unsqueeze(torch.FloatTensor(x),0).to(device)  # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        # 设置探索机制
        if np.random.uniform()< EPSILON: #np.random.uniform() 生成一个 0 到 1 之间的随机数，判断是否小于探索率 EPSILON。
            # 若小于设定值，则采用Q中的最优方法

            # 即 通过评估网络，得到输出max的那一个动作

            action_value = self.eval_net(x) # 通过评估网络得到动作值
            # 选定action
            action = torch.max(action_value,1)[1].data.cpu().numpy() # 输出每一行最大值的索引，并转化为numpy ndarray形式
            action = action[0]
        else:
            action = np.random.randint(0,N_action) # 从动作空间中随机选择一个动作
 
        return action
 
    """
    def store_transition() 
    用于将每个时间步的经验（包括当前状态 s、选择的动作 a、获得的奖励 r 和下一个状态 s_）存储到记忆库中
    """
    def store_transition(self,s,a,r,s_):
        transition = np.hstack((s,[a,r],s_))   # 因为action和reward就只是个值不是列表，所以要在外面套个[] np.hstack()将这些值水平拼接成一个一维数组
        # 如果记忆满了需要覆盖旧的数据
        index = self.memory_counter % MEMORY_CAPACITY   # 确定在buffer中的行数
        self.memory[index,:]=transition        # 用新的数据覆盖之前的之前
        self.memory_counter +=1 # 增加记忆量计数，表示记忆库中存储的经验数量
    """
    def lean() 用于执行深度 Q 网络的学习过程，包括目标网络的更新和经验回放
    """
    def learn(self):
        # 目标网络更新，就是target network
        # self.learn_step_counter 达到目标网络更新的频率 TARGET_REPLACE_ITER，则执行目标网络更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            # 讲实时网络的参数，copy给目标网络
            self.target_net.load_state_dict(self.eval_net.state_dict())   # 将实时网络的权重参数赋给目标网络
        self.learn_step_counter +=1                 # 目标函数的学习次数+1 (保持目标网络与评估网络一致)
 
        # 抽buffer中的数据学习 (经验回放)
        sample_idex = np.random.choice(MEMORY_CAPACITY,BATCH_SIZE)   # 在[0, 2000)内随机抽取32个数，可能会重复,若更改超参数会变更
        b_memory = self.memory[sample_idex,:]    # 抽取选中的行数的数据
 
        # 抽取出32个s数据，保存入b_s(buffer_state)中
        b_s = torch.FloatTensor(b_memory[:,:N_state]).to(device)
        # 抽取出32个a数据，保存入b_a中
        b_a = torch.LongTensor(b_memory[:,N_state:N_state+1]).to(device)
        # 抽取出32个r数据，保存入b_r中
        b_r = torch.FloatTensor(b_memory[:,N_state+1:N_state+2]).to(device)
        # 抽取出32个s_数据，保存入b_s_中
        b_s_ = torch.FloatTensor(b_memory[:,-N_state:]).to(device)
 
        # 获得32个trasition的评估值和目标值，并利用损失函数和优化器进行实时网络参数更新
        q_eval = self.eval_net(b_s).gather(1, b_a)         # 因为已经确定在s时候所走的action，因此选定该action对应的Q值
        # q_next 不进行反向传播，故用detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_next = self.target_net(b_s_).detach()
        # 先算出目标值q_target，max(1)[0]相当于计算出每一行中的最大值（注意不是上面的索引了,而是一个一维张量了），view()函数让其变成(32,1)
        q_target = b_r + GAMMA*q_next.max(1)[0].view(BATCH_SIZE,1)
        # 计算loss
        loss = self.loss_func(q_eval,q_target)
        self.optimazer.zero_grad()# 清空上一步的残余更新参数值
        loss.backward() # 误差方向传播
        self.optimazer.step() # 逐步的梯度优化
 
dqn= DQN()
 
# 初始化一个空列表存储每轮episode的奖励
episode_rewards = []
# 记录训练开始时间
start_time = time.time()

 
for i in range(500):                    # 设置400个episode
    print(f"<<<<<<<<< Episode{i+1}")
    s,_ = env.reset()                    # 重置环境
    episode_reward_sum = 0              # 初始化每个周期的reward值
 
    while True:
        env.render()                    # 开启画面
        a = dqn.choose_action(s)        # 与环境互动选择action
        s_,r,done, info,_= env.step(a)
        #下一个动作，奖励，是否结束，其他信息
 
        """
        可以修改reward值让其训练速度加快
        在这个例子中，通过对水平位置和倾斜角度的惩罚，可以加速学习过程，
        促使智能体更快地学到在环境中保持杆子平衡的策略
        """
        x, x_dot, theta, theta_dot = s_ # 从下一个状态 s_ 中提取的特征值（x, x_dot, theta, theta_dot）
        # r1 和 r2 是根据这些特征值计算得到的两个调整后的奖励
        # r1 是关于水平位置 x 的奖励，该奖励在智能体离目标水平位置较远时较大，随着智能体靠近目标水平位置而减小
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # r2 是关于倾斜角度 theta 的奖励，该奖励在智能体的杆子倾斜角度较大时较大，随着倾斜角度减小而减小
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        new_r = r1 + r2
 
        ########
        dqn.store_transition(s,a,new_r,s_)  # 储存样本，以便后续从中随机抽取样本进行训练
        episode_reward_sum += r # 记录当前 episode的累计奖励值
 
        s = s_                          # 进入下一个状态
 
        if dqn.memory_counter > MEMORY_CAPACITY:   # 只有在buffer中存满了数据才会学习
            dqn.learn()
 
        if done: # 检查当前 episode 是否已经结束
            """
            如果当前 episode 结束
            将当前 episode 的累计奖励值 episode_reward_sum 添加到 episode_rewards 列表中
            这个列表用于记录每个 episode 的累计奖励，以便后续绘制训练进展图
            """
            episode_rewards.append(episode_reward_sum)
            print(f"episode:{i+1},reward_sum:{episode_reward_sum}")
            break
 
# 记录结束时间
end_time = time.time()
# 计算训练总时间
training_time = end_time - start_time
print(f"Total training time: {training_time} seconds")
 
 
# 画出reward曲线图
"""
使用 plt.plot 函数绘制训练过程中每个 episode 的累计奖励值曲线
横坐标是 episode 的序号，纵坐标是累计奖励值
"""
plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, label='Episode Rewards')
plt.xlabel('Episode') # 设置横坐标的标签为 "Episode"
plt.ylabel('Total Reward') # 设置纵坐标的标签为 "Total Reward"
plt.title('Training Progress') # 设置图表的标题为 "Training Progress"
plt.legend() # 显示图例，图例标签为 'Episode Rewards'
#plt.show()
plt.savefig('./dqn.png')
 
# 关闭环境
env.close()