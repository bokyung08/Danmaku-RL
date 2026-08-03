import torch
from torch import nn


class NatureCNN(nn.Module):
    """Atari Nature CNN. (B, C, 84, 84) uint8 -> (B, output_size)"""

    def __init__(self, input_shape, output_size, hidden_size=512):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),  # 84 -> 20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),              # 20 -> 9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),              # 9  -> 7
            nn.ReLU(),
            nn.Flatten(),
        )
        # flatten 크기를 하드코딩하지 않는다 (frame stack 수가 바뀌어도 동작).
        with torch.no_grad():
            n_flatten = self.conv(torch.zeros(1, *input_shape)).shape[1]

        self.head = nn.Sequential(
            nn.Linear(n_flatten, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        # buffer는 uint8로 저장하고 float 변환은 여기서 한 번만 한다.
        # 배경은 render.py에서 이미 0으로 그려지므로, 0~255를 0~1로
        # 맞추는 것만으로 배경 픽셀은 정확히 0이 된다.
        return self.head(self.conv(x.float() / 255.0))


class MLP(nn.Module):
    """벡터 관측용. DanmakuVecEnv를 쓰게 될 경우를 위해 남겨둔다."""

    def __init__(self, input_shape, output_size, hidden_size=128):
        super().__init__()
        input_size = 1
        for dimension in input_shape:
            input_size *= dimension
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.network(x.float())
