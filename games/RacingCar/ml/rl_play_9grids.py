import numpy as np
from .QT import QLearningTable

class MLPlay:
    def __init__(self):
        self.observations = np.array([i for i in range(9)])
        
        self.reward = 0
        self.action = 1

        self.action_space = [["BRAKE"],["SPEED"],["MOVE_LEFT"],["MOVE_RIGHT"]]
        self.n_actions = len(self.action_space)
        self.RL = QLearningTable(actions=list(range(self.n_actions)))
        
        self.status = "GAME_ALIVE"
        self.state = self.observations
        self.state_ = self.observations
        

    def update(self, scene_info: dict,*args,**kwargs):
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"
        
        # 9 grids 示意圖: x 為自身車的位置
        #  (c = 0) (c = 1) (c = 2)
        # |   0   |   1   |   2   |
        # -------------------------
        # |   3   |   x   |   5   |
        # -------------------------
        # |   6   |   7   |   8   |
        def neighbor_check():
            self.observations = [0,0,0,0,0,0,0,0,0]
            x = scene_info["all_cars_pos"][0][0]  # scene_info["all_cars_pos"]: 場上所有車座標位置，[0]: user車輛座標 -> 其中[0]: x 座標
            y = scene_info["all_cars_pos"][0][1]  # [1]: y 座標
            cars_pos = scene_info["all_cars_pos"]
            user_car_grid_x = (x//120) * 120  # user car 九宮格每一格大小 改為 x 長 120 
            user_car_grid_y = (y//50) * 50  # y 寬 50
            for i, car in enumerate(cars_pos):
                corner = [(car[0], car[1]), (car[0],car[1]+30), (car[0]+60, car[1]), (car[0]+60, car[1]+30)]  # 車子四個角座標 -> [0]: x / [1]: y
                for pair in corner:
                    c = -1  # c 用來表示 車子所在 column
                    car_x = pair[0]
                    car_y = pair[1]
                    if car_x < 0:
                        continue
                    if car_x < user_car_grid_x:
                        if (user_car_grid_x - car_x)//120 == 0: # 在車子上一 column
                            c = 0
                    else:
                        if (car_x - user_car_grid_x)//120 == 0: # 在車子同一 column
                            c = 1
                        elif (car_x - user_car_grid_x)//120 == 1: # 在車子下一 column
                            c = 2

                    if c == 0 or c == 1 or c == 2:
                        if car_y < user_car_grid_y:
                            if (user_car_grid_y - car_y)//50 == 0: # 在車子後一 row
                                self.observations[c] = 1
                        else:
                            if (car_y - user_car_grid_y)//50 == 0: # 在車子同一 row
                                self.observations[c+3] = 1
                            elif (car_y - user_car_grid_y)//50 == 1: # 在車子前一 row
                                self.observations[c+6] = 1
                    else:
                        continue
                    
            if(y<=150):  # 在第一車道，設上方圍欄處皆有車(避免撞牆)
                for i in range(3):
                    self.observations[i] = 1
            elif(y>=500):  # 在最後一車道，設上方圍欄處皆有車(避免撞牆)
                for i in range(6,9):
                    self.observations[i] = 1
            # print(self.observations)

        def set_reward(self, state):
            """
            自行更改reward 在這邊
            """
            self.reward = 0
            action = self.action
            # print(state)
            if state[5] == 1: # 前有車
                if state[1] == 1: # 左有車
                    if state[7] == 1: # 右有車
                        if action == 0:
                            self.reward += 1
                        if action == 1:
                            self.reward -= 1
                        if action == 2:
                            self.reward -= 1
                        if action == 3:
                            self.reward -= 1
                    else: # 右沒車
                        if action == 0:
                            self.reward += 1
                        if action == 1:
                            self.reward -= 1
                        if action == 2:
                            self.reward -= 1
                        if action == 3:
                            self.reward += 5
                else: # 左沒車
                    if state[7] == 1: # 右有車
                        if action == 0:
                            self.reward += 1
                        if action == 1:
                            self.reward -= 1
                        if action == 2:
                            self.reward += 5
                        if action == 3:
                            self.reward -= 1
                    else: # 右沒車
                        if action == 0:
                            self.reward += 1
                        if action == 1:
                            self.reward -= 1
                        if action == 2:
                            self.reward += 5
                        if action == 3:
                            self.reward += 5
            else: # 前沒車
                if action == 0:
                    self.reward -= 5
                if action == 1:
                    self.reward += 5
                if action == 2:
                    self.reward -= 5
                if action == 3:
                    self.reward -= 5
            return self.reward
        
        neighbor_check()
        self.state_ = self.observations
        self.reward = set_reward(self,self.state_)
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
        self.RL.q_table.to_pickle('games/RacingCar/log/log.pickle')
        pass
     