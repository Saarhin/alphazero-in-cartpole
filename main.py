import typing

from argparse import ArgumentParser
import os

import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
import numpy as np
import random
from datetime import datetime

from core.pretrain import pretrain
from core.train import train
from core.test import test
from config.place import Config

def set_seed(seed):
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
if __name__ == "__main__":
    parser = ArgumentParser("MCTS Place, GO")
    parser.add_argument("--env", type=str, default="Place-v0")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--opr", default="train", type=str)
    parser.add_argument("--num_rollout_workers", default=4, type=int)
    parser.add_argument("--num_cpus_per_worker", default=4, type=float)
    parser.add_argument("--num_gpus_per_worker", default=0.2, type=float)
    parser.add_argument("--num_test_episodes", default=200, type=float)
    parser.add_argument("--model_path", default=None)
    # parser.add_argument(
    #     "--model_path",
    #     default="/home/swang848/efficientalphazero/results/cartpole_14082024_1540/model_latest.pt",
    # )
    parser.add_argument("--device_workers", default="cuda", type=str)
    parser.add_argument("--device_trainer", default="cuda", type=str)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--cc", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--group_name", default="default", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_target_blocks", default=5, type=int)
    parser.add_argument("--c_init", default=5.25, type=int)
    parser.add_argument("--num_simulations", default=15, type=int)
    parser.add_argument("--min_num_episodes_per_worker", default=10, type=int)
    parser.add_argument("--training_steps", default=15, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    args = parser.parse_args()
    
    set_seed(args.seed)

    sub_dir = datetime.now().strftime("%d%m%Y_%H%M")
    sub_dir = f"{args.env}_{sub_dir}"
    # if program is run on CC, save logs to the local disk.
    if args.cc:
        log_dir = f"{os.environ['results']}/{sub_dir}"
    else:
        if args.debug:
            sub_dir = f"debug/{sub_dir}"
        if os.path.isabs(args.results_dir):
            log_dir = os.path.join(args.results_dir, sub_dir)
        else:
            log_dir = os.path.join(os.getcwd(), args.results_dir, sub_dir)
        
    summary_writer = SummaryWriter(log_dir, flush_secs=10)

    config = Config(
        log_dir=log_dir
    )  # Apply set BaseConfig arguments
    
    for arg, arg_val in vars(args).items():
        if hasattr(config, arg):
            setattr(config, arg, arg_val)
            print(f'Overwriting "{arg}" config entry with {arg_val}')
        else:
            setattr(config, arg, arg_val)
            print(f'Adding "{arg}" config entry with {arg_val}')
        
    setattr(config, 'replay_buffer_size', args.num_rollout_workers * config.min_num_episodes_per_worker * config.num_target_blocks * 4)
    print(f'Overwriting "replay_buffer_size" config entry with {args.num_rollout_workers * config.min_num_episodes_per_worker * config.num_target_blocks * 4}')
            
    print(args)
            
    if args.wandb and not args.debug:
        wandb.init(project="MCTSplace", group=args.group_name, config=config)

    model = config.init_model(args.device_trainer, args.amp)  # Create (and load) model
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))

    opr_lst = args.opr.split(",")
    for opr in opr_lst:
        if opr == "train":
            train(args, config, model, summary_writer, log_dir)
        elif opr == "test":
            test(args, config, model, log_dir)
        elif opr == "pretrain":
            pretrain(args, config, model, summary_writer, log_dir)

    print("Finished")
