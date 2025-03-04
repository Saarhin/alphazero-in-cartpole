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
model_path = "/home/swang848/efficientalphazero/saved_weights/15b/Place-v0_30112024_1246_59/model_latest.pt"
model = config.init_model("cuda", True)
model.load_state_dict(torch.load(model_path))

# optimal 15b: 49, 51, 81, 40, 60, 30, 41, 31, 19, 70, 52, 29, 62, 63, 53
# a = [-0.9885300705882353, -0.9404873647058818, -0.7703811647058817, -0.7926083176470587, -0.6915497058823524, 
#  -0.6859977999999999, -0.6568149529411764, -0.666212517647059, -0.6443696823529412, -0.4795973294117648, 
# -0.47648878823529417, -0.26184647058823507, -0.1823906117647055, -0.07857954117647026, -0.07857954117647026]

# a = [-9.885300705882353, -9.404873647058817, -7.703811647058817, -7.926083176470588, -6.915497058823524, 
# -6.859977999999999, -6.568149529411764, -6.662125176470591, -6.443696823529413, -4.795973294117648, 
# -4.764887882352942, -2.618464705882351, -1.8239061176470557, -0.7857954117647026, -0.7857954117647026]

# suboptimal 15b: 40, 63, 81, 51, 40, 19, 30, 30, 29, 106, 97, 52, 40, 31, 97
# b = [-0.993702988235294, -0.9540352588235296, -0.7854679411764705, -0.7418032941176476, -0.736146764705883, 
#  -0.7216924823529419, -0.5876121058823531, -0.5703404705882357, -0.5672965764705885, -0.6364776000000005, 
#  -0.5645019058823538, -0.3840411647058828, -0.35910280000000006, -0.2966495999999993, -0.2494982235294123]

# suboptimal 15b: 53, 49, 100, 51, 60, 30, 49, 49, 29, 70, 52, 63, 81, 40, 63
# b = [-10, -10, -8.994232235294112, -8.647340823529406, -8.133417058823525, -7.446811764705885, 
# -7.358730352941176, -7.403592823529415, -7.237828823529417, -6.278181176470587, -6.333534470588233, 
# -4.947343529411761, -4.475993058823523, -3.365897294117638, -3.118038117647048]

# print(sum(a))
# print(sum(b))
action_list = [53, 49, 100, 51, 60, 30, 49, 49, 29, 70, 52, 63, 81, 40, 63]
mcts_windows = [MCTSRollingWindow(config.obs_shape, config.frame_stack)]
obs, info = env.reset()
mcts_windows[0].add(obs=obs["board_image"], env_state=env.get_state(), info=info)
order = env.place_order

print(order)

rewards = list()
for i in range(num_target_blocks):
    # use random action
    one_indices = [
        index for index, value in enumerate(info["action_mask"]) if value == 1
    ]
    # sampled_action = random.choice(one_indices)
    # use optimized action
    # sampled_action = optimized_action[order[i]]
    # print(sampled_action)
    sampled_action = action_list[i]
    priors, values = model.compute_priors_and_values(mcts_windows)
    obs, reward, done, info = env.step(sampled_action)
    mcts_windows[0].add(obs=obs["board_image"], env_state=env.get_state(), reward=reward, action=sampled_action, info=info)
    rewards.append(reward)
    print(priors, values)
    # print(info["hpwl"])
    # print(info["wirelength"])

# print(rewards)

# for i in reversed(range(num_target_blocks)):
#     value_target = value_target*0.997 + rewards[i]
#     print(value_target)
