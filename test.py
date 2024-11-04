import place_env
import gym
import random
from core.util import trans_coordinate

log_dir = "/home/swang848/efficientalphazero/test"
num_target_blocks = 5
env = gym.make("Place-v0", log_dir=log_dir, simulator=True, num_target_blocks=num_target_blocks)

optimized_action = list()
with open("/home/swang848/efficientalphazero/data/optimized.place", "r") as file:
    for index, line in enumerate(file.readlines()):
        if 60 >= index >= 5:
            line_split = line.strip().split()
            coord = trans_coordinate([int(line_split[1]), int(line_split[2])], 11, "cs")
            optimized_action.append(coord[0] * 11 + coord[1])

obs, infos = env.reset()
order = env.place_order

rewards = list()
for i in range(num_target_blocks):
    # use random action
    one_indices = [index for index, value in enumerate(infos['action_mask']) if value == 1]
    sampled_action = random.choice(one_indices)
    # use optimized action
    sampled_action = optimized_action[order[i]]
    print(sampled_action)
    obs, reward, done, infos = env.step(sampled_action)
    rewards.append(reward)
    print(infos['hpwl'])
    print(infos['wirelength'])
    

# for i in reversed(range(num_target_blocks)):
#     value_target = value_target*0.997 + rewards[i]
#     print(value_target)