import gym
import os
import place_env
import time


log_dir = os.path.join(os.getcwd(), "results/debug")
env = gym.make("Place-v0", log_dir=log_dir, simulator=True, num_target_blocks=5)

action_list = [52, 30, 29, 40, 19]
obs, infos = env.reset()
for action in action_list:
    start_time = time.time()
    obs, reward, done, infos = env.step(action)
    end_time = time.time()
    print(reward)
    print(end_time - start_time)
    