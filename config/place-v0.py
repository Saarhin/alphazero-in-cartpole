import os
import gym

from config.base import BaseConfig
from core.model import ResModel
from core.util import DiscreteSupport

from core.preprocess import Preporcess
import place_env


class Config(BaseConfig):
   
    def __init__(
        self,
        log_dir: str,
        training_steps: int = 20,
        pretrain_steps: int = 0,
        model_broadcast_interval: int = 5,
        num_sgd_iter: int = 10,
        clear_buffer_after_broadcast: bool = False,
        root_value_targets: bool = False,
        replay_buffer_size: int = 50000,
        demo_buffer_size: int = 0,
        batch_size: int = 512,
        lr: float = 1e-3,
        max_grad_norm: float = 5,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        c_init: float = 3.0,
        c_base: float = 19652,
        gamma: float = 0.997,
        frame_stack: int = 5,
        max_reward_return: bool = False,
        hash_nodes: bool = False,
        root_dirichlet_alpha: float = 1.5,
        root_exploration_fraction: float = 0.25,
        num_simulations: int = 30,
        num_envs_per_worker: int = 1,
        min_num_episodes_per_worker: int = 2,
        # min_num_episodes_per_worker: int = 8,
        use_dirichlet: bool = True,
        test_use_dirichlet: bool = False,
        value_support: DiscreteSupport = DiscreteSupport(0, 22, 1.0),
        value_transform: bool = True,
        env_seed: int = None,
    ):
        super().__init__(
            training_steps,
            pretrain_steps,
            model_broadcast_interval,
            num_sgd_iter,
            clear_buffer_after_broadcast,
            root_value_targets,
            replay_buffer_size,
            demo_buffer_size,
            batch_size,
            lr,
            max_grad_norm,
            weight_decay,
            momentum,
            c_init,
            c_base,
            gamma,
            frame_stack,
            max_reward_return,
            hash_nodes,
            root_dirichlet_alpha,
            root_exploration_fraction,
            num_simulations,
            num_envs_per_worker,
            min_num_episodes_per_worker,
            use_dirichlet,
            test_use_dirichlet,
            value_support,
            value_transform,
            env_seed,
        )
        
    def init_model(self, device, amp):      
        env = self.env_creator()
        obs_shape = env.observation_space.shape
        num_act = env.action_space.n 
        
        model = ResModel(self, obs_shape, num_act, device, amp)
        model.to(device)
        return model, env
    
    def env_creator(self, log_dir):
        return gym.make("Place-v0", log_dir)
        