import numpy as np
from .Qtable import QLearningTable

class MLPlay:
    def __init__(self):
        self.car_pos = (20, 160)
        self.coin_num = 0
        self.coin_num_ = 0
        self.cars_pos = []
        self.coins_pos = []
        self.car_lane = self.car_pos[1] // 50  # lanes 1 ~ 9
        self.lanes = [110, 160, 210, 260, 310, 360, 410, 460, 510] # lanes center
        self.step = 0
        self.reward = 0
        self.action = 0
        self.action_space = []
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.RL = QLearningTable(actions=list(range(self.n_actions)))        
        self.observation = 0
        self.coin = 0
        self.status = "ALIVE"
        self.state = []
        self.state_ = []

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

            # 

        def step(self, state):
            # reward function
            self.reward = 0

            #

            return self.reward


        self.car_pos = (scene_info["x"], scene_info["y"])
        self.cars_pos = scene_info["all_cars_pos"]
        self.car_vel = scene_info["velocity"]
        self.car_dis = scene_info["distance"]
        
        self.coin_num = self.coin_num_

        check_grid()
        self.state_ = []
        self.reward = step(self, self.state_)
        action = self.RL.choose_action(str(self.state))
        self.RL.learn(str(self.state), self.action, self.reward, str(self.state_))
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
    #    self.RL.q_table.to_pickle('games/RacingCar/log/qlearning.pickle')
    #    self.RL.plot_cost()
        print("reset ml script")
        
        pass
