import numpy as np
from .QT_normal import QLearningTable

class MLPlay:
    def __init__(self):
        self.car_vel = 0
        self.car_pos = (20, 160)
        self.car_dis = 0
        self.coin_num = 0
        self.coin_num_ = 0
        self.cars_pos = []
        self.coins_pos = []
        self.car_lane = self.car_pos[1] // 50  # lanes 1 ~ 9
        self.lanes = [110, 160, 210, 260, 310, 360, 410, 460, 510] # lanes center
        self.step = 0
        self.reward = 0
        self.action = 0
        self.action_space = [["SPEED", "MOVE_LEFT"], ["SPEED", "MOVE_RIGHT"], ["MOVE_LEFT"], ["MOVE_RIGHT"], ["SPEED"], ["BRAKE"]]
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.RL = QLearningTable(actions=list(range(self.n_actions)))        
        self.observation = 0
        self.coin = 0
        self.status = "ALIVE"
        self.state = [self.observation, self.coin_num]
        self.state_ = [self.observation, self.coin_num]

        print("Initial ml script")


    # 16 grid relative position  

#      |    |    |    |    |    |
#      |  1 |  2 |  3 |  4 |  5 |
#      |    |    |    |    |    |
#      |  6 |  c |  8 |  9 | 10 |
#      |    |    |    |    |    |
#      | 11 | 12 | 13 | 14 | 15 |
#      |    |    |    |    |    |


    def update(self, scene_info: dict):
        """
        Generate the command according to the received scene information
        """
        if scene_info["status"] == "GAME_OVER":
            return "RESET"

        self.car_pos = (scene_info["x"], scene_info["y"])
        self.cars_pos = scene_info["all_cars_pos"]
        self.car_vel = scene_info["velocity"]
        self.car_dis = scene_info["distance"]

        def check_grid():
            self.observation = 0

            # car position
            x = scene_info["x"]
            y = scene_info["y"]

            if y <= 140: # lane 1
                self.observation = 1
            if y >= 490: # lane 9
                self.observation = 2            

            for i in range(len(self.cars_pos)):
                (zi,wi) = self.cars_pos[i]

            #    if (x, y) != (zi, wi):
                if y == wi:  # 前後車
                    if -120 < x - zi <= -90:
                        self.observation = 3
                    if -90 < x - zi <= -60:
                        self.observation = 4
                    if -60 < x - zi <= -30:
                        self.observation = 5
                    if -30 < x - zi < 0:
                        self.observation = 6
                    if 0 < x - zi <= 30:
                        self.observation = 7
                    if 30 < x - zi <= 60:
                        self.observation = 8
                    if 60 < x - zi <= 90:
                        self.observation = 9
                
                if (-30 < y - wi < 30) and (y != wi):  # 前後車
                    if -120 < x - zi <= -90:
                        self.observation = 10
                    if -90 < x - zi <= -60:
                        self.observation = 11
                    if -60 < x - zi <= -30:
                        self.observation = 12
                    if -30 < x - zi < 0:
                        self.observation = 13
                    if 0 < x - zi <= 30:
                        self.observation = 14
                    if 30 < x - zi <= 60:
                        self.observation = 15
                    if 60 < x - zi <= 90:
                        self.observation = 16
                
                if -90 < x - zi < 90:  # 左右車
                    if 5 <= y - wi <= 75:
                        self.observation = 17
                    if -75 <= y - wi <= -5:
                        self.observation = 18
                
                if y - wi < -50 or y - wi > 50:
                    self.observation = 19
                
            #    if -45 <= y - wi <= 45 and -75 <= x - zi <= 75:
            #        self.observation = 20


        def step(self, state):
            # reward function
            self.reward = 0
            
            # STATE
            if state[0] == 0:
                if self.action == 5:
                    self.reward -= 1000
                if self.action == 4:
                    self.reward += 500
            if state[0] == 1:
                if self.action == 0:
                    self.reward -= 1000
                if self.action == 2:
                    self.reward -= 1000
            if state[0] == 2:
                if self.action == 1:
                    self.reward -= 1000
                if self.action == 3:
                    self.reward -= 1000
            if state[0] == 3:
                if self.action == 1:
                    self.reward += 500
                if self.action == 2:
                    self.reward += 500
                if self.action == 3:
                    self.reward += 500
                if self.action == 4:
                    self.reward += 500
            if state[0] == 4:
                if self.action == 1:
                    self.reward += 800
                if self.action == 2:
                    self.reward += 800
                if self.action == 3:
                    self.reward += 800
                if self.action == 4:
                    self.reward += 800
            if state[0] == 5:
                if self.action == 4:
                    self.reward -= 1000
                if self.action == 5:
                    self.reward += 1000
            if state[0] == 6:
                if self.action == 4:
                    self.reward -= 1000
                if self.action == 5:
                    self.reward += 1000
            if state[0] == 7:
                if self.action == 5:
                    self.reward -= 1000
            if state[0] == 8:
                if self.action == 5:
                    self.reward -= 1000
            if state[0] == 12:
                if self.action == 4:
                    self.reward -= 1000
                if self.action == 5:
                    self.reward += 1000
            if state[0] == 13:
                if self.action == 4:
                    self.reward -= 1000
                if self.action == 5:
                    self.reward += 1000
            if state[0] == 14:
                if self.action == 5:
                    self.reward -= 1000
            if state[0] == 15:
                if self.action == 5:
                    self.reward -= 1000
            if state[0] == 17:
                if self.action == 0:
                    self.reward -= 1000
                if self.action == 2:
                    self.reward -= 1000
            if state[0] == 18:
                if self.action == 1:
                    self.reward -= 1000
                if self.action == 3:
                    self.reward -= 1000  
            if state[0] == 19:
                if self.action == 5 :
                    self.reward -= 800
                if self.action == 3:
                    self.reward -= 800
                if self.action == 2:
                    self.reward -= 800
                if self.action == 4:
                    self.reward += 1000

            # velocity
        #    if  self.car_vel <= 1.2:
        #        self.reward -= 10
        #    if 1.2 < self.car_vel <= 5:
        #        self.reward += 10
        #    if 5 < self.car_vel <= 10:
        #        self.reward += 20
        #    if 10 < self.car_vel <= 15:
        #        self.reward += 30
            
            # distance
            if self.car_dis <= 100:
                self.reward += 10
            if 100 < self.car_dis <= 1500:
                self.reward += 50
            if 1500 < self.car_dis <= 5000:
                self.reward += 500    
            if 5000 < self.car_dis <= 8000:
                self.reward += 1000
            if 8000 < self.car_dis <= 10000:
                self.reward += 1500
            if 10000 < self.car_dis <= 15000:
                self.reward += 3000

            return self.reward


        self.car_pos = (scene_info["x"], scene_info["y"])
        self.cars_pos = scene_info["all_cars_pos"]
        self.car_vel = scene_info["velocity"]
        self.car_dis = scene_info["distance"]
        
        self.coin_num = self.coin_num_

        check_grid()
        self.state_ = [self.observation, self.coin_num_ - self.coin_num]
        self.reward = step(self, self.state_)
    #    action = 4
        action = self.RL.choose_action(str(self.state))
        self.RL.learn(str(self.state), self.action, self.reward, str(self.state_))
    #    print(str(self.state), self.action, self.reward, str(self.state_))
        self.action = action
        self.state = self.state_
       
        if scene_info["status"] != "GAME_ALIVE" and scene_info["status"] != "ALIVE":
            return "RESET"

        if scene_info.__contains__("coin"):
            self.coin_pos = scene_info["coin"]
        
        return self.action_space[action]


    def reset(self):
        """
        Reset the status
        """
        print(self.RL.q_table)
        self.coin_num = 0
        self.RL.q_table.to_pickle('games/RacingCar/log/qlearning.pickle')
    #    self.RL.plot_cost()
        print("reset ml script")
        
        pass
