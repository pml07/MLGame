import numpy as np
from .QT import QLearningTable

class MLPlay:
    def __init__(self):
        self.car_vel = 0
        self.car_pos = (0, 160)
        self.car_dis = 0
        self.coin_num = 0
        self.coin_num_ = 0
        self.cars_pos = []
        self.coins_pos = []
        self.car_lane = self.car_pos[1] // 50  # lanes 1 ~ 9
        self.lanes = [125, 175, 225, 275, 325, 375, 425, 475, 525] # lanes center
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
        self.state = [self.observation, self.coin_num, self.status, self.reward]
        self.state_ = [self.observation, self.coin_num, self.status, self.reward]

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

        self.car_pos = (scene_info["x"], scene_info["y"])
        self.cars_pos = scene_info["all_cars_pos"]
        self.car_vel = scene_info["velocity"]
        self.car_dis = scene_info["distance"]

        def check_grid():
            self.observation = 0
            grid = set()
            if self.car_pos[1] <= 125: # lane 1
                grid.add(1)
                grid.add(2)
                grid.add(3)
                grid.add(4)
                grid.add(5)
                self.observation = 1
            if self.car_pos[1] >= 525: # lane 9
                grid.add(11)
                grid.add(12)
                grid.add(13)
                grid.add(14)
                grid.add(15)
                self.observation = 2


            for i in range(len(self.cars_pos)):
                # car position
                x = scene_info["x"]
                y = scene_info["y"]
            #    print((x,y))
                
                (xi,yi) = self.cars_pos[i]
            #    print((xi, yi))

                if x < 0:  # 過起點
                    if y == yi:  # 同車道前後車
                        if -180 < abs(x) - xi <= -150:
                            grid.add(10)
                            self.observation = 3
                        if -150 < abs(x) - xi <= -120:
                            grid.add(9)
                            self.observation = 4
                        if -120 < abs(x) - xi <= -60:
                            grid.add(8)
                            self.observation = 5
                        if -60 < abs(x) - xi <= 0:
                            grid.add(7)
                            self.observation = 6
                        if 60 < abs(x) - xi <= 120:
                            grid.add(6)
                            self.observation = 7

                    if -30 < abs(x) - xi < 30:  # 左右車
                        if y - yi == 50:
                            grid.add(1)
                            grid.add(2)
                            self.observation = 8
                        if y - yi == -50:
                            grid.add(11)
                            grid.add(12)
                            self.observation = 9
                # 未過起點
                else: 
                    return ["SPEED"]


        def step(self, state):
            # reward function
            self.reward = 0
            self.car_vel = scene_info["velocity"]
            self.car_dis = scene_info["distance"]

            # state
            if state[2] == "GAME_ALIVE" :
                self.reward += 500
            elif state[2] == "GAME_OVER":
                self.reward -= 10000
            elif state[2] == "GAME_PASS":
                self.reward += 10000

            # observation
            if state[0] == 3:
                self.reward += 30
            if state[0] == 4:
                self.reward += 50
            if state[0] == 5:
                self.reward += 10
            if state[0] == 6:
                self.reward -= 200
            if state[0] == 7:
                self.reward += 20
            
            # velocity
            if  self.car_vel <= 4:
                self.reward -= 1000
            if 4 < self.car_vel <= 9:
                self.reward += 500
            if 9 < self.car_vel <= 15:
                self.reward += 1000
            
            # distance
            if self.car_dis <= 1000:
                self.reward += 50
            if 1000 < self.car_dis <= 5000:
                self.reward += 100
            if 5000 < self.car_dis <= 10000:
                self.reward += 5000
            if 10000 < self.car_dis <= 15000:
                self.reward += 1000
            if self.car_dis == 14900:
                self.reward += 5000

            return self.reward
        
        self.car_pos = (scene_info["x"], scene_info["y"])
        self.cars_pos = scene_info["all_cars_pos"]
        self.car_vel = scene_info["velocity"]
        self.car_dis = scene_info["distance"]
        
        self.coin_num = self.coin_num_

        check_grid()
        self.state_ = [self.observation, self.coin_num_ - self.coin_num, scene_info["status"], self.reward]
        self.reward = step(self, self.state_)
        action = self.RL.choose_action(str(self.state))
        self.RL.learn(str(self.state), self.action, self.reward, str(self.state_))
    #    print(str(self.state), self.action, self.reward, str(self.state_))
        self.action = action
        self.state = self.state_
       
        if scene_info["status"] != "GAME_ALIVE" and scene_info["status"] != "ALIVE":
            return "RESET"
        
        return self.action_space[action]


    def reset(self):
        """
        Reset the status
        """
        print(self.RL.q_table)
        self.coin_num = 0
        self.RL.q_table.to_pickle('games/RacingCar/log/table0.pickle')
    #    self.RL.plot_cost()
        print("reset ml script")
        
        pass
