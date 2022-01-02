import numpy as np
import random

class MLPlay:
    def __init__(self):
        self.other_cars_position = []
        self.coins_pos = []
        print("Initial ml script")

        self.neighbor = np.zeros(25, dtype=int)
        self.neighbor_array = np.zeros(25, dtype=int)
        self.command = "SPEED"
        self.previous_command = "SPEED"
    
        
    
    def neighbor_check(self, x, y, cars_pos):
        self.neighbor_array = np.zeros(25, dtype=int)
        new_x = (x//120) * 120 # makes car position is right at the top left corner of the grid
        new_y = (y//50) * 50 ### - 15
        for i, car in enumerate(cars_pos):
            neighbor = [(car[0], car[1]), (car[0],car[1]+30), (car[0]+60, car[1]), (car[0]+60, car[1]+30)]
            for pair in neighbor:
                c = -1 # represents the column where it stays
                n_x = pair[0]
                n_y = pair[1]
                if n_x < 0:
                    continue
                if n_x < new_x:
                    if (new_x - n_x)//120 == 0: # last column
                        c = 0
                else:
                    if (n_x - new_x)//120 == 0: # same column
                        c = 1
                    elif (n_x - new_x)//120 == 1: # next column
                        c = 2
                    elif (n_x - new_x)//120 == 2: # next of next column
                        c = 3
                    elif (n_x - new_x)//120 == 3:
                        c = 4

                if c == 0 or c == 1 or c == 2 or c == 3 or c == 4:
                    if n_y < new_y:
                        if (new_y - n_y)//50 == 0: # last row
                            self.neighbor_array[c+5] = 1
                        elif (new_y-n_y)//50 == 1: # last of last row
                            self.neighbor_array[c] = 1
                    else:
                        if (n_y - new_y)//50 == 0: # same row
                            self.neighbor_array[c+10] = 1
                        elif (n_y - new_y)//50 == 1: # next row
                            self.neighbor_array[c+15] = 1
                        elif (n_y - new_y)//50 == 2:
                            self.neighbor_array[c+20] = 1
                else:
                    continue

        if(y<=150): #在第一車道及最後一車道避免撞牆
            for i in range(5):
                self.neighbor_array[i] = 1
        elif(y>=500):
            for i in range(20,25):
                self.neighbor_array[i] = 1

        return self.neighbor_array


    def update(self, scene_info: dict):
        """
        Generate the command according to the received scene information
        """
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"

        if scene_info.__contains__("coin"):
            self.coin_pos = scene_info["coin"]
        
        self.car_pos = (scene_info["x"], scene_info["y"])
        self.other_cars_position = scene_info["all_cars_pos"]
        
        self.neighbor = self.neighbor_check(self.car_pos[0],self.car_pos[1],self.other_cars_position)

        if (self.neighbor[12] == 1 ): #or self.neighbor[13] == 1
            self.command = random.choice((["MOVE_RIGHT"],["MOVE_LEFT"]))
            if self.previous_command == ["MOVE_RIGHT"]:
                self.previous_command = ["MOVE_RIGHT"]
                return["MOVE_RIGHT"]
            elif self.previous_command == ["MOVE_LEFT"]:
                self.previous_command = ["MOVE_LEFT"]
                return["MOVE_LEFT"]
            else:
                self.previous_command = self.command
                return self.command

        else: 
            self.previous_command = ["SPEED"]
            return ["SPEED"]



    def reset(self):
        """
        Reset the status
        """
        print("reset ml script")
        pass