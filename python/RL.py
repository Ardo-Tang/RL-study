import gym
from gym import envs
import numpy as np

class RL(object):

    def __init__(self, env_name="Pong-v0", learning_rate=0.01):
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.learning_rate = learning_rate
        
    def step(self, action):
        state, reward, game_over, _ = self.env.step(action)
        # observation：環境狀態，像是影像 pixel，角度，角速度等，參數意義必須參照原始碼
        # reward：越好的結果，會回饋越大的值，若不適用，也可從 observation 自行產生
        # done：判斷是否達到 game over 的條件，像是生命已結束，或是已經超出範圍
        # info：debug 用的資訊，不允許使用在學習上
        return state, reward, game_over
    
    