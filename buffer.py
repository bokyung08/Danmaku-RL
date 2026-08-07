import numpy as np
import torch
from typing import NamedTuple


# data = ReplayBufferSamples(...) 일 때
# data.rewards 와 같이 꺼내올 수 있음
class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor

class ReplayBuffer:
    def __init__(
            self,
            capacity: int,
            obs_shape,
            device
    ):
        self.capacity = capacity
        self.obs_shape = tuple(obs_shape)
        self.device = torch.device(device)
        self.data_type = torch.uint8 if len(obs_shape) > 2 else torch.float32

        # 이미지 관측은 uint8로 저장한다. float32로 저장하면 메모리가 4배가 되고,
        # 어차피 model.forward()에서 한 번 변환하므로 여기서는 uint8을 유지한다.
        # 메모리 = capacity * 2 * prod(obs_shape) 바이트
        self.observations = torch.empty((capacity, *self.obs_shape), dtype=self.data_type)
        self.next_observations = torch.empty((capacity, *self.obs_shape), dtype=self.data_type)
        self.actions = torch.empty((capacity, 1), dtype=torch.long)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32)
        self.dones = torch.empty((capacity, 1), dtype=torch.float32)

        self.pos = 0  # 저장할 현재 위치
        self.size = 0  # 현재 버퍼 크기

    def __len__(self):
        return self.size

    def _as_obs_tensor(self, obs):
        # 이미지 관측(uint8)과 벡터 관측(float32, [-1,1])을 self.data_type에 맞는
        # numpy dtype으로 캐스팅한다. 항상 uint8로 캐스팅하면 벡터 관측이 망가진다.
        # deque에서 np.stack된 배열이라 연속 메모리를 보장하기 위해 ascontiguousarray를 거친다.
        numpy_dtype = np.uint8 if self.data_type == torch.uint8 else np.float32
        return torch.from_numpy(np.ascontiguousarray(obs, dtype=numpy_dtype))

    @torch.no_grad()
    def add(self, transition):
        obs, action, reward, terminated, next_obs = transition

        self.observations[self.pos] = self._as_obs_tensor(obs)
        self.next_observations[self.pos] = self._as_obs_tensor(next_obs)
        self.actions[self.pos, 0] = int(action)
        self.rewards[self.pos, 0] = float(reward)
        self.dones[self.pos, 0] = float(terminated)

        # buffer 정보 업데이트
        self.pos = (self.pos+1) % self.capacity
        self.size = min(self.capacity, self.size+1)

    def sample(self, batch_size):
        if self.size < batch_size:
            raise RuntimeError("더 충분한 데이터를 모으고 sample하세요")

        indices = torch.randint(
            low=0, high=self.size, size=(batch_size,),
        )

        ob = self.observations[indices].to(self.device)
        a = self.actions[indices].to(self.device)
        r = self.rewards[indices].to(self.device)
        next_ob = self.next_observations[indices].to(self.device)
        d = self.dones[indices].to(self.device)

        return ReplayBufferSamples(
            observations=ob,
            actions=a,
            rewards=r,
            next_observations=next_ob,
            dones=d
        )

        