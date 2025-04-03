import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Network adapted from: https://github.com/LukeDitria/pytorch_tutorials/blob/main/section11_rl/solutions/Procgen_PPO.ipynb


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x_in):
        x = self.relu(x_in)
        x = F.relu(self.conv1(x))
        x_out = self.conv2(x)
        return x_in + x_out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, block_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, block_channels, 3, 1, 1)
        self.max_pool = nn.MaxPool2d(3, 2, 1)
        self.res1 = ResBlock(block_channels)
        self.res2 = ResBlock(block_channels)

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.max_pool(x)
        x = self.res1(x)
        return self.res2(x)


class ImpalaCNN64(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        base_channels: int = 16,
    ):
        super().__init__(observation_space, features_dim)

        # determine number of input channels
        in_channels = observation_space.shape[0]  # e.g. 4 with FrameStack(4)

        self.conv = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(2 * base_channels, 2 * base_channels),
            nn.Flatten(),
        )

        # Compute CNN output dim
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            sample_output = self.conv(sample_input)
            cnn_output_dim = sample_output.shape[1]

        self.linear = nn.Sequential(nn.Linear(cnn_output_dim, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor):
        return self.linear(self.conv(observations))
