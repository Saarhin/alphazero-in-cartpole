import place_env
import gym
import random
from core.util import trans_coordinate
import torch
from config.place import Config
from core.replay_buffer import MCTSRollingWindow

log_dir = "/home/swang848/efficientalphazero/test"
num_target_blocks = 15
env = gym.make(
    "Place-v0", log_dir=log_dir, simulator=False, num_target_blocks=num_target_blocks
)

optimized_action = list()
with open("/home/swang848/efficientalphazero/data/optimized.place", "r") as file:
    for index, line in enumerate(file.readlines()):
        if 60 >= index >= 5:
            line_split = line.strip().split()
            coord = trans_coordinate([int(line_split[1]), int(line_split[2])], 11, "cs")
            optimized_action.append(coord[0] * 11 + coord[1])
            
config = Config(log_dir=log_dir)
model_path = "/home/swang848/efficientalphazero/saved_weights/15b/Place-v0_25112024_1811/model_latest.pt"
model = config.init_model("cuda", True)
model.load_state_dict(torch.load(model_path))

mcts_windows = [MCTSRollingWindow(config.obs_shape, config.frame_stack)]
obs, infos = env.reset()
order = env.place_order

rewards = list()
for i in range(num_target_blocks):
    # use random action
    one_indices = [
        index for index, value in enumerate(infos["action_mask"]) if value == 1
    ]
    sampled_action = random.choice(one_indices)
    # use optimized action
    # sampled_action = optimized_action[order[i]]
    print(sampled_action)
    priors, values = model.compute_priors_and_values(mcts_windows)
    obs, reward, done, infos = env.step(sampled_action)
    mcts_windows[0].add(obs=obs, env_state=env.get_state(), reward=reward, action=sampled_action, infos=infos)
    rewards.append(reward)
    print(infos["hpwl"])
    print(infos["wirelength"])

print(rewards)

# for i in reversed(range(num_target_blocks)):
#     value_target = value_target*0.997 + rewards[i]
#     print(value_target)
