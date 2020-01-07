import numpy as np
import pandas as pd
import time

class FindT:
    State_space = 6 #狀態

    Actions = ['left','right'] #動作
    Actions_space = 2 #動作空間

    learning_rate = 0.1 #學習率
    memory = 0.9 #記憶擇優比重

    gamma = 0.9 #衰減常數

    epochs = 20

    Q_table = pd.DataFrame([])
    
    def __init__(self, State_space = 6, Actions = ['left','right'], learning_rate = 0.1, gamma = 0.9, episodes = 20):
        self.State_space = State_space
        self.Actions = Actions
        self.Actions_space = len(Actions)
        self.learning_rate = learning_rate
        self.memory = 1-learning_rate
        self.gamma = gamma
        self.epochs = episodes
        self.Q_table = self.__init_Q_table()
    
    def __init_Q_table(self):
        table = pd.DataFrame(np.zeros((self.State_space, self.Actions_space)), columns = self.Actions)
        return table

    def __choose_action(self, state):
        get_reward = self.Q_table.iloc[state, :] #依據當前state, 從Q_table中找action
        
        if(np.random.uniform()<self.learning_rate or get_reward.all()==0):# 隨機小於learning rate或get_reward中沒有reward時學習
            action_name = np.random.choice(self.Actions)
        else:
            action_name = get_reward.argmax()

        return action_name
    
if __name__ == '__main__':
    game = FindT()
    print(game.Q_table)
