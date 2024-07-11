from snake import *
from utility import *
from cube import *

import pygame
import numpy as np
from tkinter import messagebox
from snake import Snake
import matplotlib.pyplot as plt



def main():
    rewards = []
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))

    snake_1 = Snake((255, 0, 0), (15, 15))
    snake_2 = Snake((255, 255, 0), (5, 5))  # dont forget the file names later 
    snake_1.addCube()
    snake_2.addCube()

    snack = Cube(randomSnack(ROWS, snake_1), color=(0, 255, 0))

    clock = pygame.time.Clock()

    while True:
        reward_1 = 0
        reward_2 = 0
        pygame.time.delay(50)
        clock.tick(500)
        
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                if messagebox.askokcancel("Quit", "Do you want to save the Q-tables?"):
                    save(snake_1, snake_2)
                pygame.quit()
                exit()
                
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                np.save("plot_train.npy", snake_1.q_table)
                np.save("plot_train.npy", snake_2.q_table)
                np.save("rewards.npy", rewards)
                pygame.time.delay(1000)

        state_1, new_state_1, action_1 = snake_1.move(snack, snake_2)
        state_2, new_state_2, action_2 = snake_2.move(snack, snake_1)

        snack, reward_1, win_1, win_2 = snake_1.calc_reward(snack, snake_2)
        snack, reward_2, win_2, win_1 = snake_2.calc_reward(snack, snake_1)

        snake_1.update_q_table(state_1, action_1, new_state_1, reward_1)
        snake_2.update_q_table(state_2, action_2, new_state_2, reward_2)
        rewards.append((reward_1, reward_2))

        snake_1.decay_epsilon()
        snake_2.decay_epsilon()
        print(reward_1, reward_2)
        
        redrawWindow(snake_1, snake_2, snack, win)


def plot_rewards():
    rewards = np.load("rewards.npy")
    rewards_1 = [r[0] for r in rewards]
    rewards_2 = [r[1] for r in rewards]
    plt.plot(rewards_1, label='Snake 1 Rewards')
    plt.plot(rewards_2, label='Snake 2 Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards Over Time')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
