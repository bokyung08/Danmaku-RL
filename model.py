import torch
from torch import nn

from env import DanmakuVecEnv


def _head_desc(hidden_size, output_size, dueling_net):
    """describe()들이 공유하는 마지막 Q-head 설명 (dueling 여부에 따라 갈림)."""
    if dueling_net:
        return (
            f"dueling: V=Linear({hidden_size}->1), "
            f"A=Linear({hidden_size}->{output_size}), Q=V+A-mean(A)"
        )
    return f"Linear({hidden_size}->{output_size})"


class NatureCNN(nn.Module):
    """Atari Nature CNN. (B, C, 84, 84) uint8 -> (B, output_size)"""

    def __init__(
        self,
        input_shape,
        output_size,
        hidden_size=512,
        layer_norm=False,
        dueling_net=False,
    ):
        super().__init__()
        self.dueling_net = dueling_net
        self.layer_norm = layer_norm
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),  # 84 -> 20
            nn.LayerNorm([32, 20, 20]) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),              # 20 -> 9
            nn.LayerNorm([64, 9, 9]) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),              # 9  -> 7
            nn.LayerNorm([64, 7, 7]) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Flatten(),
        )
        # flatten 크기를 하드코딩하지 않는다 (frame stack 수가 바뀌어도 동작).
        with torch.no_grad():
            self.n_flatten = self.conv(torch.zeros(1, *input_shape)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(self.n_flatten, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ReLU(),
        )
        if self.dueling_net:
            self.value = nn.Linear(hidden_size, 1)
            self.advantage = nn.Linear(hidden_size, output_size)
        else:
            self.final = nn.Linear(hidden_size, output_size)

    def forward_features(self, x):
        """Return the penultimate representation used by the Q-value layer."""
        encoded = self.conv(x.float() / 255.0)
        return self.linear(encoded)

    def forward(self, x):
        features = self.forward_features(x)
        if not self.dueling_net:
            return self.final(features)

        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def describe(self):
        ln = " -> LayerNorm" if self.layer_norm else ""
        lines = [
            f"[conv1]  Conv2d({self.conv[0].in_channels}->{self.conv[0].out_channels},"
            f"k={self.conv[0].kernel_size[0]},s={self.conv[0].stride[0]}){ln} -> ReLU",
            f"[conv2]  Conv2d({self.conv[3].in_channels}->{self.conv[3].out_channels},"
            f"k={self.conv[3].kernel_size[0]},s={self.conv[3].stride[0]}){ln} -> ReLU",
            f"[conv3]  Conv2d({self.conv[6].in_channels}->{self.conv[6].out_channels},"
            f"k={self.conv[6].kernel_size[0]},s={self.conv[6].stride[0]}){ln} -> ReLU",
            f"[fc]     Flatten -> Linear({self.n_flatten}->{self.hidden_size}){ln} -> ReLU",
            f"[head]   {_head_desc(self.hidden_size, self.output_size, self.dueling_net)}",
        ]
        return "\n            ".join(lines)


class MLP(nn.Module):
    """벡터 관측용 (DanmakuVecEnv)."""

    def __init__(
        self,
        input_shape,
        output_size,
        hidden_size=128,
        layer_norm=False,
        dueling_net=False,
    ):
        super().__init__()
        self.dueling_net = dueling_net
        self.layer_norm = layer_norm
        self.hidden_size = hidden_size
        self.output_size = output_size
        input_size = 1
        for dimension in input_shape:
            input_size *= dimension
        self.input_size = input_size
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ReLU(),
        )
        if self.dueling_net:
            self.value = nn.Linear(hidden_size, 1)
            self.advantage = nn.Linear(hidden_size, output_size)
        else:
            self.final = nn.Linear(hidden_size, output_size)

    def forward_features(self, x):
        return self.linear(x.float())

    def forward(self, x):
        features = self.forward_features(x)
        if not self.dueling_net:
            return self.final(features)

        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def describe(self):
        ln = " -> LayerNorm" if self.layer_norm else ""
        lines = [
            f"[fc1]    Flatten -> Linear({self.input_size}->{self.hidden_size}){ln} -> ReLU",
            f"[fc2]    Linear({self.hidden_size}->{self.hidden_size}){ln} -> ReLU",
            f"[head]   {_head_desc(self.hidden_size, self.output_size, self.dueling_net)}",
        ]
        return "\n            ".join(lines)


class ScaledDotProductAttention(nn.Module):
    """Score = QK^T / sqrt(d_k), softmax(Score) 를 V에 곱한다."""

    def forward(self, q, k, v, mask=None):
        scores = q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5)
        if mask is not None:
            # (B, seq) -> (B, 1, 1, seq) 로 head/query 축에 broadcast
            while mask.dim() < scores.dim():
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return weights @ v, weights


class AttentionQNetwork(nn.Module):
    """attention for vec env
    agent를 query로, 공들을 key/value로 두는 multi-head cross-attention
    num_heads=1이면 single-head attention으로 작동
    """

    def __init__(
        self,
        input_shape,
        output_size,
        hidden_size=512,
        layer_norm=False,
        dueling_net=False,
        num_heads=4,
        fusion_mode="residual",
        position_mode="relative",
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.dueling_net = dueling_net
        self.layer_norm = layer_norm
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.fusion_mode = fusion_mode  # "residual" or "concat"
        self.position_mode = position_mode  # "relative" or "absolute"

        self.agent_n = DanmakuVecEnv.AGENT_FEATURE_NUM
        self.ball_n = DanmakuVecEnv.BALL_FEATURE_NUM
        self.max_balls = (input_shape[0] - self.agent_n) // self.ball_n

        # linear -> (layernorm) -> relu block
        def make_block(in_size):
            return nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
                nn.ReLU(),
            )

        self.agent_encoder = make_block(self.agent_n)
        # mask(ball_feat의 마지막 값)는 넣지 않음. 어차피 attention에서 key, value 제외 용도로만 사용
        self.ball_encoder = make_block(self.ball_n - 1)  # (4,256) -> (256, 256)

        # 매 episode 시작처럼 공이 하나도 없을 때(=모든 슬롯이 mask=0)도 attention이
        # 참조할 대상이 하나는 있어야 하므로, 항상 마스킹되지 않는 "공 없음"을 나타내는
        # 학습 가능한 key/value를 하나 추가해둔다 (안 그러면 모든 key가 마스킹된
        # softmax 입력이 전부 -inf가 되어 NaN이 난다).
        self.empty_kv = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.attention = ScaledDotProductAttention()
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.combine_norm = nn.LayerNorm(hidden_size) if layer_norm else nn.Identity()
        if fusion_mode == "concat":
            self.fuse = nn.Linear(hidden_size * 2, hidden_size)

        if self.dueling_net:
            self.value = nn.Linear(hidden_size, 1)
            self.advantage = nn.Linear(hidden_size, output_size)
        else:
            self.final = nn.Linear(hidden_size, output_size)

    def _split_heads(self, x):
        # (batch, seq, hidden) -> (batch, num_heads, seq, head_dim)
        batch_size, seq_length, _ = x.shape
        x = x.view(batch_size, seq_length, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        # (batch, num_heads, seq, head_dim) -> (batch, seq, hidden)
        batch_size, _, seq_length, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_length, self.hidden_size)

    def forward_features(self, x):
        x = x.float()
        batch_size = x.shape[0]
        agent_feat = x[:, :self.agent_n]
        ball_feat = x[:, self.agent_n:].reshape(batch_size, self.max_balls, self.ball_n)  # (B, 40, 5)
        ball_xy = ball_feat[..., :2]
        ball_v = ball_feat[..., 2:4]
        ball_mask = ball_feat[..., 4]  # (B, max_balls), 1=real, 0=padding

        if self.position_mode == "relative":
            ball_xy = ball_xy - agent_feat[:, :2].unsqueeze(1)
        ball_input = torch.cat([ball_xy, ball_v], dim=-1)  # (B, max_balls, 4)

        agent_embed = self.agent_encoder(agent_feat)  # (B, hidden)
        ball_embed = self.ball_encoder(ball_input)     # (B, max_balls, hidden)

        empty_kv = self.empty_kv.expand(batch_size, 1, self.hidden_size)
        kv = torch.cat([ball_embed, empty_kv], dim=1)  # (B, max_balls+1, hidden)
        sentinel_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        key_padding_mask = torch.cat([ball_mask == 0, sentinel_mask], dim=1)  # True=제외

        q = self._split_heads(self.q_linear(agent_embed).unsqueeze(1))  # (B, heads, 1, head_dim)
        k = self._split_heads(self.k_linear(kv))
        v = self._split_heads(self.v_linear(kv))

        attn_out, _ = self.attention(q, k, v, mask=key_padding_mask)
        attn_out = self._merge_heads(attn_out).squeeze(1)  # (B, hidden)
        attn_out = self.out_proj(attn_out)

        if self.fusion_mode == "concat":
            # 차원이 늘어났기 때문에 fuse (linear)로 차원을 줄임
            combined = self.fuse(torch.cat([agent_embed, attn_out], dim=-1))
        else:
            combined = agent_embed + attn_out
        combined = self.combine_norm(combined)
        return torch.relu(combined)

    def forward(self, x):
        features = self.forward_features(x)
        if not self.dueling_net:
            return self.final(features)

        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def describe(self):
        """agent/ball encoder가 병렬로 처리된 뒤 attention -> combine으로 합쳐지는
        구조라 순차 나열("A -> B -> C")로는 구조가 안 드러난다. 학습 로그에서
        실제 흐름이 보이도록 직접 사람이 읽을 설명을 만든다.
        """
        ln = " -> LayerNorm" if self.layer_norm else ""
        head_desc = _head_desc(self.hidden_size, self.output_size, self.dueling_net)
        ball_input_size = self.ball_n - 1
        ball_feature_desc = "x,y,vx,vy"
        position_desc = "agent 기준 상대좌표" if self.position_mode == "relative" else "화면 기준 절대좌표"
        projection_desc = f"Linear({self.hidden_size}->{self.hidden_size})"
        if self.fusion_mode == "concat":
            combine_desc = (
                f"concat(agent embed, attention) -> Linear({self.hidden_size * 2}->"
                f"{self.hidden_size}){ln} -> ReLU"
            )
        else:
            combine_desc = f"agent embed + attention{ln} -> ReLU"
        lines = [
            f"[agent encoder]     Linear({self.agent_n}->{self.hidden_size}){ln} -> ReLU",
            f"[ball encoder]      Linear({ball_input_size}->{self.hidden_size}){ln} -> ReLU"
            f"  ({ball_feature_desc}; x,y는 {position_desc}."
            f" 공 {self.max_balls}개에 각각 동일하게 적용, 파라미터 공유)",
            f"[+ sentinel]        학습되는 '공 없음' key/value 1개 추가 (항상 마스킹 안 함)",
            f"[cross-attention]   Q=agent embed(1개) , K=V=ball embed({self.max_balls}개)+sentinel(1개)"
            f" -> {self.num_heads} head x {self.head_dim}dim -> softmax(QK^T/sqrt(d))·V"
            f" (mask=0인 빈 슬롯은 attention에서 제외)",
            f"[out_proj]          {projection_desc}",
            f"[combine]           {combine_desc}",
            f"[head]              {head_desc}",
        ]
        return "\n            ".join(lines)
