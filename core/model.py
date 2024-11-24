from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config.base import BaseConfig
from core.replay_buffer import MCTSRollingWindow, TrainingBatch


class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class ResNet(nn.Module):
    def __init__(self, input_shape, num_blocks=10, out_channels=64, hidden_size=512):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], out_channels, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(out_channels, out_channels) for _ in range(num_blocks)]
        )
        self._hidden_size = hidden_size
        
        self.conv2 = nn.Conv2d(out_channels, out_channels // 4, kernel_size=1)
        self.obs_out = nn.Linear(input_shape[1] * input_shape[2] * out_channels // 4, hidden_size)

        self.train()

    def forward(self, x):
        x = self.conv1(x)

        for block in self.res_blocks:
            x = block(x)

        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten
        obs_out = F.relu(self.obs_out(x))

        return obs_out

    @property
    def output_size(self):
        return self._hidden_size


class BaseModel(nn.Module):
    def __init__(self, config, obs_shape, num_act, device, amp):
        super().__init__()
        self.config: BaseConfig = config
        self.obs_shape = obs_shape
        self.num_act = num_act
        self.device = device
        self.amp = amp

    def forward(self):
        raise NotImplementedError

    def compute_priors_and_values(self):
        raise NotImplementedError

    def update_weights(self):
        raise NotImplementedError

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)


class ResModel(BaseModel):
    def __init__(self, config, obs_shape, num_act, device, amp, hidden_size=512):
        super().__init__(config, obs_shape, num_act, device, amp)
        
        self.shared = ResNet(obs_shape, num_blocks=10, out_channels=64, hidden_size=hidden_size)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_act),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, config.value_support.size),
        )
        
        self.to(device)

    def forward(self, x):
        # Cuda only supports bfloat16 computation in ampere or newer GPUS 
        # TODO: CPU only support torch.bfloat16 in autocast, but we are only doing experiment on GPU so no need to accomodate CPU
        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.amp):
            feat = self.shared(x)
            p = self.actor(feat)
            v = self.critic(feat)

        return p, v

    def compute_priors_and_values(self, windows: List[MCTSRollingWindow]):
        obs = np.stack([window.obs for window in windows])
        obs = torch.from_numpy(obs).to(self.device).float()     
        mask = np.stack([window.infos[0]["action_mask"] for window in windows])
        mask = torch.from_numpy(mask).to(self.device).bool()

        with torch.no_grad():
            policy_logits, values_logits = self.forward(obs)

        masked_policy_logits = torch.where(mask, policy_logits, torch.tensor(-float('inf')).to(self.device))
        priors = nn.Softmax(dim=-1)(masked_policy_logits)
        values_softmax = nn.Softmax(dim=-1)(values_logits)
        values = self.config.phi_inverse_transform(values_softmax).flatten()

        if self.config.value_transform:
            values = self.config.inverse_scalar_transform(values)

        priors = priors.cpu().float().numpy()
        values = values.cpu().float().numpy()
        return priors, values

    def update_weights(self, train_batch, optimizer, scaler, scheduler):
        self.train()
        train_batch.to_torch(self.device)

        policy_logits, value_logits = self.forward(train_batch.obs)
        masked_policy_logits = torch.where(train_batch.action_mask, policy_logits, torch.tensor(-float('inf')).to(self.device))

        value_targets = train_batch.value_targets
        if self.config.value_transform:
            value_targets = self.config.scalar_transform(value_targets)
        value_targets_phi = self.config.phi_transform(value_targets)

        policy_loss = -(torch.log_softmax(masked_policy_logits, dim=1) * train_batch.mcts_policies)
        infinite_value_mask = torch.isfinite(policy_loss).to(self.device)
        policy_loss = policy_loss[infinite_value_mask].mean()
        value_loss = -(torch.log_softmax(value_logits, dim=1) * value_targets_phi).mean()

        # Update prios
        """
        values_pred = self.config.phi_inverse_transform(value_logits)
        with torch.no_grad():
            new_priorities = nn.L1Loss(reduction='none')(values_pred, value_targets).cpu().numpy().flatten()
        new_priorities += 1e-5
        replay_buffer.update_priorities.remote(batch_indices, new_priorities)
        """
        parameters = self.parameters()

        total_loss = (policy_loss + value_loss) / 2
        scaler.scale(total_loss).backward()
        torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        return total_loss, policy_loss, value_loss