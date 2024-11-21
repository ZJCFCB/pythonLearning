import numpy as np  
import os
import random

#这里的Q-learning算法实习的是一个走迷宫的游戏。迷宫如图Q-1.jpg所示

#随机生成一个动作
def random_action(V):
	index_list = []
	for index, s in enumerate(list(V)):
		if s >= 0:
			index_list.append(index)
	return random.choice(index_list)

#Reward Setting
def reward_setting(state_num, action_num):
	R = -1 * np.ones((state_num , action_num))
	R[0,4] = 0
	R[1,3] = 0
	R[1,5] = 100
	R[2,3] = 0
	R[3,1] = 0
	R[3,2] = 0
	R[3,4] = 0
	R[4,0] = 0 
	R[4,3] = 0
	R[4,5] = 100
	R[5,1] = 0
	R[5,4] = 0
	R[5,5] = 100
	return R 


if __name__ == '__main__':
	action_num = 6
	state_num = 6
	gamma = 0.8
	alpha = 0.8
	epoch_number = 1000
	condition_stop = 5

	Q = np.zeros((state_num , action_num))
	R = reward_setting(state_num , action_num)

	for epoch in range(epoch_number):
		for s in range(state_num):
			loop = True
			while loop:
				#从s出发选择动作，来到了状态a（可以这样理解)
				a = random_action(R[s,:])
				# 求一下状态a的q_max
				q_max = np.max(Q[a,:]) 
				# 更新从s到a的q值
				#Q[s,a] = R[s,a] + gamma * q_max
				Q[s,a] = Q[s,a] + alpha*(R[s,a]+gamma*q_max - Q[s,a])
				# 从状态s继续出发，直到终点
				s = a
				if s == condition_stop:
					loop = False
	Q = (Q / 5).astype(int)
	print(Q)
