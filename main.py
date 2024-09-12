import typing

from argparse import ArgumentParser
import os

import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
from datetime import datetime

from core.pretrain import pretrain
from core.train import train
from core.test import test
from config.base import BaseConfig

if __name__ == "__main__":
    print("Go!")
    parser = ArgumentParser("AlphaZero implemented efficiently using Ray.")
    parser.add_argument("--env", type=str, default="Place-v0")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--opr", default="test", type=str)
    parser.add_argument("--num_rollout_workers", default=4, type=int)
    # parser.add_argument("--num_rollout_workers", default=1, type=int)
    parser.add_argument("--num_cpus", default=8, type=float)
    parser.add_argument("--num_gpus", default=0, type=float)
    parser.add_argument("--num_cpus_per_worker", default=4, type=float)
    parser.add_argument("--num_gpus_per_worker", default=0.25, type=float)
    parser.add_argument("--num_test_episodes", default=200, type=float)
    parser.add_argument("--model_path", default=None)
    # parser.add_argument(
    #     "--model_path",
    #     default="/home/swang848/efficientalphazero/results/cartpole_14082024_1540/model_latest.pt",
    # )
    parser.add_argument("--device_workers", default="cpu", type=str)
    parser.add_argument("--device_trainer", default="cpu", type=str)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--wandb", default=False, type=bool)
    parser.add_argument("--debug", default=True, type=bool)
    parser.add_argument("--group_name", default="default", type=str)

    config_args = (
        []
    )  # Add config.base.BaseConfig constructor parameters to ArgumentParser
    for arg, type_hint in typing.get_type_hints(BaseConfig.__init__).items():
        if type_hint not in [int, float, str, bool]:
            continue
        parser.add_argument(f"--{arg}", default=None, required=False, type=type_hint)
        config_args.append(arg)
    args = parser.parse_args()

    if args.env == "Place-v0":
        from config.place import Config
    else:
        raise ValueError
    
    sub_dir = datetime.now().strftime("%d%m%Y_%H%M")
    sub_dir = f"{args.env}_{sub_dir}"
    if args.debug:
        sub_dir = f"debug/{sub_dir}"
    log_dir = os.path.join(args.results_dir, sub_dir)
    summary_writer = SummaryWriter(log_dir, flush_secs=10)
    
    config = Config(env_seed=args.env_seed, log_dir=log_dir)  # Apply set BaseConfig arguments
    for arg in config_args:
        arg_val = getattr(args, arg)
        if getattr(args, arg) is not None:
            # TODO: use_dirichlet (or bool arguments in general) do not seem to work properly
            setattr(config, arg, arg_val)
            print(f'Overwriting "{arg}" config entry with {arg_val}')

    if not args.debug and args.wandb:
        wandb.init(project="AlphaZero_Cartpole", group=args.group_name, config=config)
    else:
        pass

    model = config.init_model(
        args.device_trainer, args.amp, log_dir
    )  # Create (and load) model
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
