from cube import Cube
from constants import *
from utility import *

import random
import numpy as np

class Snake:
    body = []
    turns = {}

    def __init__(self, color, pos, file_name=None):
        self.color = color
        self.head = Cube(pos, color=color)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        self.last_dirnx = 0
        self.last_dirny = 1
        self.prev_action = 0
        self.break_flag = 0

        try:
            self.q_table = np.load(file_name)
        except:
            self.q_table = np.zeros((4, 4, 4, 4))

        self.lr = 0.1 
        self.discount_factor = 0.95
        self.epsilon = 1 
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def get_direction(self, pos1, pos2):
        if pos1[0] < pos2[0]:
            return 0  
        elif pos1[0] > pos2[0]:
            return 1  
        elif pos1[1] < pos2[1]:
            return 2  
        else:
            return 3  

    def get_danger(self):
        head_pos = self.head.pos
        danger = [0, 0, 0, 0] 

        if head_pos[0] <= 0 or (head_pos[0] - 1, head_pos[1]) in list(map(lambda z: z.pos, self.body)):
            danger[0] = 1 
        if head_pos[0] >= ROWS - 1 or (head_pos[0] + 1, head_pos[1]) in list(map(lambda z: z.pos, self.body)):
            danger[1] = 1 
        if head_pos[1] <= 0 or (head_pos[0], head_pos[1] - 1) in list(map(lambda z: z.pos, self.body)):
            danger[2] = 1 
        if head_pos[1] >= ROWS - 1 or (head_pos[0], head_pos[1] + 1) in list(map(lambda z: z.pos, self.body)):
            danger[3] = 1  

        return danger.index(1) if 1 in danger else -1

    def get_state(self, snack, other_snake):
        direction_snack = self.get_direction(self.head.pos, snack.pos)
        direction_other_snake = self.get_direction(self.head.pos, other_snake.head.pos)
        direction_danger = self.get_danger()
        return (direction_snack, direction_other_snake, direction_danger)

    def get_optimal_policy(self, state):
        return np.argmax(self.q_table[state[0], state[1], state[2], :])

    def make_action(self, state):
        chance = random.random()
        if chance < self.epsilon:
            action = random.randint(0, 3)
        else:
            action = self.get_optimal_policy(state)
        return action

    def update_q_table(self, state, action, next_state, reward):
        max_future_q = np.max(self.q_table[next_state[0], next_state[1], next_state[2], :])
        current_q = self.q_table[state[0], state[1], state[2], action]
        new_q = (1 - self.lr) * current_q + self.lr * (reward + self.discount_factor * max_future_q)
        self.q_table[state[0], state[1], state[2], action] = new_q
        self.decay_epsilon()

    def check_movement_loop_and_keep_going(self, action):
        if action == 0 and self.last_dirny == 1:
            self.break_flag = 1
            return 1
        elif action == 1 and self.last_dirny == -1:
            self.break_flag = 1
            return 0
        elif action == 2 and self.last_dirnx == 1:
            self.break_flag = 1
            return 3
        elif action == 3 and self.last_dirnx == -1:
            self.break_flag = 1
            return 2
        return action
    
    def check_movement_loop_and_turn(self, action):
        if (self.dirnx == 1 and action == 2) or (self.dirnx == -1 and action == 3) or\
            (self.dirny == 1 and action == 0) or (self.dirny == -1 and action == 1):
                self.break_flag = 1
                return random.choice([2, 3])
        return action
        
    def move(self, snack, other_snake):
        state = self.get_state(snack, other_snake)
        action = self.make_action(state)

        temp_action1 = self.check_movement_loop_and_keep_going(action)
        temp_action2 = self.check_movement_loop_and_turn(action)
        if(self.break_flag):
            action = random.choice([temp_action1, temp_action2])

        if action == 0:  
            self.dirny = -1
            self.dirnx = 0
            self.last_dirnx = self.dirnx
            self.last_dirny = self.dirny
        elif action == 1: 
            self.dirny = 1
            self.dirnx = 0
            self.last_dirnx = self.dirnx
            self.last_dirny = self.dirny
        elif action == 2:  
            self.dirnx = -1
            self.dirny = 0
            self.last_dirnx = self.dirnx
            self.last_dirny = self.dirny
        elif action == 3:  
            self.dirnx = 1
            self.dirny = 0
            self.last_dirnx = self.dirnx
            self.last_dirny = self.dirny

        self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)

        self.break_flag = 0
        next_state = self.get_state(snack, other_snake)
        return state, next_state, action

    def check_out_of_board(self):
        headPos = self.head.pos
        if headPos[0] >= ROWS - 1 or headPos[0] < 1 or headPos[1] >= ROWS - 1 or headPos[1] < 1:
            self.reset((random.randint(3, 18), random.randint(3, 18)))
            return True
        return False

    def calc_reward(self, snack, other_snake):
        reward = 0
        win_self, win_other = False, False

        if self.check_out_of_board():
            reward -= 30 
            win_other = True
            self.reset((random.randint(3, 18), random.randint(3, 18)))

        if self.head.pos == snack.pos:
            self.addCube()
            snack = Cube(randomSnack(ROWS, self), color=(0, 255, 0))
            reward += 30  
        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            reward -= 10  
            win_other = True
            self.reset((random.randint(3, 18), random.randint(3, 18)))

        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):
            if self.head.pos != other_snake.head.pos:
                reward -= 20  
                win_other = True
            else:
                if len(self.body) > len(other_snake.body):
                    reward += 40  
                    win_self = True
                elif len(self.body) == len(other_snake.body):
                    reward = 0 
                else:
                    reward -= 20  
                    win_other = True
            self.reset((random.randint(3, 18), random.randint(3, 18)))

        return snack, reward, win_self, win_other

    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1]), color=self.color))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1]), color=self.color))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1), color=self.color))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1), color=self.color))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)

    def save_q_table(self, file_name):
        np.save(file_name, self.q_table)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
