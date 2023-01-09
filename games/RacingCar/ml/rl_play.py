import numpy as np
from .QT import QLearningTable

class MLPlay:
    def __init__(self):
        self.observation = 0
        self.reward = 0
        self.action = 1

        self.action_space = [["BRAKE"],["SPEED"],["MOVE_LEFT"],["MOVE_RIGHT"]]
        self.n_actions = len(self.action_space)
        self.RL = QLearningTable(actions=list(range(self.n_actions)))
        
        self.status = "GAME_ALIVE"
        self.state = [self.observation, self.action]
        self.state_ = [self.observation, self.action]

        self.front = 0
        self.top = 0
        self.down = 0
        self.length = -45
        self.position = 0
        self.target = 15000
        self.done = 0
        #print(self.RL.q_table)
 
    def update(self, scene_info: dict,*args,**kwargs):
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"

        def check():        # 設定observation
            self.observation = 0
            self.front = 0
            self.top = 0
            self.down = 0
            self.target = 15000
            self.done = 0
            for i in range (1,len(scene_info["all_cars_pos"])):
                if scene_info["all_cars_pos"][i][1] == ((self.position-1)*50)+100+10:    # 同賽道有車
                    if scene_info["all_cars_pos"][i][0] - scene_info["all_cars_pos"][0][0] < 210 and scene_info["all_cars_pos"][i][0] > scene_info["all_cars_pos"][0][0]:   # 前方有車
                        self.front = 1
                        break
                else:
                    self.front = 0

            if self.front == 1:  # 如果前方有車
                for j in range(1,len(scene_info["all_cars_pos"])):
                        if ((scene_info["all_cars_pos"][j][0] - scene_info["all_cars_pos"][0][0] < 180 and scene_info["all_cars_pos"][j][0] - scene_info["all_cars_pos"][0][0] > self.length)
                            and scene_info["all_cars_pos"][j][1] == (self.position-2)*50+100+10): # 車上方
                            self.top = 1
                        if ((scene_info["all_cars_pos"][j][0] - scene_info["all_cars_pos"][0][0] < 180 and scene_info["all_cars_pos"][j][0] - scene_info["all_cars_pos"][0][0] > self.length)
                            and scene_info["all_cars_pos"][j][1] == (self.position)*50+100+10): # 車下方
                            self.down = 1

            if self.position == 1:
                self.top = 1
            if self.position == 9:
                self.down = 1


            if self.front == 0:          # 前方沒車
                if scene_info["all_cars_pos"][0][1] - (((self.position-1)*50)+100+10) > 5 and scene_info["all_cars_pos"][0][1] > ((self.position-1)*50)+100+10:     # 車子在路線下方
                    self.observation = 3 
                elif scene_info["all_cars_pos"][0][1] - (((self.position-1)*50)+100+10) < -5 and scene_info["all_cars_pos"][0][1] < ((self.position-1)*50)+100+10:     # 車子在路線上方
                    self.observation = 2
                else:
                    self.observation = 1 

            elif self.front == 1 :     # 前方有車
                if self.top == 1 and self.down == 1:   # 上下前都有車
                    self.observation = 4 
                elif self.top == 1:       # 上前有車
                    self.observation = 5 
                elif self.down == 1:     # 前下有車
                    self.observation = 6 
                else:                    # 只有前面有車
                    self.observation = 7 
            else:
                pass
            

        def step(self, state):      # 設reward
            self.reward = 0
            # 0: brake / 1: speed / 2: left / 3: right
            action = self.action

            if state == [1,0]:      # 前方沒車
                self.reward -= 10
            if state == [1,1]:     
                self.reward += 10
            if state == [1,2]:      
                self.reward -= 10
            if state == [1,3]:      
                self.reward -= 10
            
            if state == [2,0]:      # 前方沒車且車子在路線上方
                self.reward -= 10
            if state == [2,1]:
                self.reward += 10
            if state == [2,2]:
                self.reward += 0
            if state == [2,3]:
                self.reward += 0
                
            if state == [3,0]:      # 前方沒車且車子在路線下方
                self.reward -= 10
            if state == [3,1]:
                self.reward += 10
            if state == [3,2]:
                self.reward += 0
            if state == [3,3]:
                self.reward += 0
                  
            if state == [4,0]:      # 上下前都有車
                self.reward += 1
            if state == [4,1]:
                self.reward -= 1
            if state == [4,2]:
                self.reward -= 1
            if state == [4,3]:
                self.reward -= 1
                
            if state == [5,0]:      # 前上有車
                self.reward += 0
            if state == [5,1]:
                self.reward -= 1
            if state == [5,2]:
                self.reward -= 1
            if state == [5,3]:
                self.reward += 10
                
            if state == [6,0]:      # 前下有車
                self.reward += 0
            if state == [6,1]:
                self.reward -= 1
            if state == [6,2]:
                self.reward += 10
            if state == [6,3]:
                self.reward -= 1
                
            if state == [7,0]:      # 只有前面有車
                self.reward += 1
            if state == [7,1]:      # 只有前面有車
                self.reward -= 10
            if state == [7,2]:      # 只有前面有車
                self.reward += 1
            if state == [7,3]:      # 只有前面有車
                self.reward += 1
            return self.reward
            
        
        #self.length = -45        # 判斷上下

        if scene_info["velocity"] < 10:
            self.length = -60
        self.position = ((scene_info["all_cars_pos"][0][1]-100)//50)+1   # 目前車在哪個賽道上
      
        check()

        self.state_ = [self.observation, self.action]
        self.reward = step(self,self.state_)
        action = self.RL.choose_action(str(self.state))
        self.RL.learn(str(self.state), self.action, self.reward, str(self.state_))
        self.action = action
        self.state = self.state_

        return self.action_space[action] 

    def reset(self):
        """
        Reset the status
        """
        print(self.RL.q_table)
        self.RL.q_table.to_pickle('games/RacingCar/log/log_6.pickle')
        pass

