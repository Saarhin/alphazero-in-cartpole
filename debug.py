import gym
import os
import place_env
import time


log_dir = os.path.join(os.getcwd(), "results/debug")
env = gym.make("Place-v0", log_dir=log_dir, simulator=False, num_target_blocks=15)

# c5b optimal
action_list = [40, 30, 19, 52, 29]
# c15b optimal
action_list = [49, 51, 81, 40, 60, 30, 41, 31, 19, 70, 52, 29, 62, 63, 53]
obs, infos = env.reset()
for action in action_list:
    obs, reward, done, infos = env.step(action)
    print(infos['placed_block'])
    print(reward)
    