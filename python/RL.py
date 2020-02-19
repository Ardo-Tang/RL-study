import gym
from gym import envs
import numpy as np
from math import log2
from keras import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, LeakyReLU

class RL(object):

    def __init__(self, env_name="Pong-v0", learning_rate=0.01):
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.learning_rate = learning_rate
        self.policy = self.mkmodel()
        
    def step(self, action):
        state, reward, game_over, _ = self.env.step(action)
        # state：環境狀態，像是影像 pixel，角度，角速度等，參數意義必須參照原始碼
        # reward：越好的結果，會回饋越大的值，若不適用，也可從 observation 自行產生
        # game_over：判斷是否達到 game over 的條件，像是生命已結束，或是已經超出範圍
        # info：debug 用的資訊，不允許使用在學習上
        return state, reward, game_over

    def mkmodel(self):
        model = Sequential()
        model.add(Convolution2D(filters=32, kernel_size=3, input_shape=self.env.observation_space.shape))
        model.add(LeakyReLU())
        model.add(Convolution2D(filters=32, kernel_size=3))
        model.add(LeakyReLU())
        model.add(MaxPooling2D(pool_size=2, padding="same"))

        model.add(Convolution2D(filters=64, kernel_size=3))
        model.add(LeakyReLU())
        model.add(Convolution2D(filters=64, kernel_size=3))
        model.add(LeakyReLU())
        model.add(MaxPooling2D(pool_size=2, padding="same"))

        model.add(Flatten())
        
        output_shape = 2 ** int(log2(model.output_shape[1]))
        while(int(output_shape/2)>=256):
            model.add(Dense(int(output_shape/2)))
            model.add(LeakyReLU())
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            output_shape = int(output_shape/2)

        model.add(Dense(self.action_space.n))
        
        model.summary()
        return model

    def train(self):
        self.policy.compile(optimizer="Nadam", loss="mse")

        

if __name__ == "__main__":
    RL = RL()
    
    pass
