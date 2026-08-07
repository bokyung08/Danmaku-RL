"""Evaluate a saved Danmaku checkpoint on a deterministic seed range."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config
from agent import evaluate
from env import DanmakuImgEnv, DanmakuVecEnv
from model import AttentionQNetwork, MLP, NatureCNN


class GreedyAgent:
    def __init__(self, model, n_actions, device):
        self.model = model
        self.n_actions = n_actions
        self.device = device
        self.epsilon = 0.0

    def get_action(self, observation):
        with torch.inference_mode():
            tensor = torch.as_tensor(observation, device=self.device).unsqueeze(0)
            return int(self.model(tensor).argmax(dim=1).item())


def load_checkpoint(checkpoint, device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = payload["model_state_dict"]
    metadata = payload.get("meta", {})

    saved_config = {}
    config_path = checkpoint.parent / "config.json"
    if config_path.exists():
        saved_config.update(json.loads(config_path.read_text(encoding="utf-8")))
    saved_config.update(metadata.get("config", {}))

    is_image = any(key.startswith("conv.") for key in state_dict)
    use_attention = bool(saved_config.get("USE_ATTENTION", False))
    hidden_size = int(saved_config.get("HIDDEN_SIZE", config.HIDDEN_SIZE))
    layer_norm = bool(saved_config.get("LAYER_NORM", False))
    dueling_net = bool(saved_config.get("DUELING_NET", False))
    frame_skip = int(saved_config.get("N_FRAME_SKIP", config.N_FRAME_SKIP))

    if is_image:
        env = DanmakuImgEnv()
        model_class = NatureCNN
        extra_kwargs = {}
    elif use_attention:
        env = DanmakuVecEnv(
            normalize=saved_config.get("VEC_OBS_NORMALIZE", "none")
        )
        model_class = AttentionQNetwork
        inferred_fusion = (
            "concat" if any(key.startswith("fuse.") for key in state_dict) else "residual"
        )
        extra_kwargs = {
            "num_heads": int(
                saved_config.get("ATTENTION_NUM_HEADS", config.ATTENTION_NUM_HEADS)
            ),
            "fusion_mode": saved_config.get("ATTENTION_FUSION", inferred_fusion),
            "position_mode": saved_config.get(
                "ATTENTION_POSITION_MODE", config.ATTENTION_POSITION_MODE
            ),
        }
    else:
        env = DanmakuVecEnv(
            normalize=saved_config.get("VEC_OBS_NORMALIZE", "near")
        )
        model_class = MLP
        extra_kwargs = {}

    model = model_class(
        env.observation_shape,
        env.n_actions,
        hidden_size=hidden_size,
        layer_norm=layer_norm,
        dueling_net=dueling_net,
        **extra_kwargs,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return GreedyAgent(model, env.n_actions, device), env, frame_skip, saved_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=30_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    agent, env, frame_skip, saved_config = load_checkpoint(args.checkpoint, device)
    original_frame_skip = config.N_FRAME_SKIP
    config.N_FRAME_SKIP = frame_skip
    try:
        metrics = evaluate(
            agent,
            env,
            num_episodes=args.episodes,
            seed=args.seed_base,
        )
    finally:
        config.N_FRAME_SKIP = original_frame_skip
        env.close()

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "seed_base": args.seed_base,
        "config": saved_config,
        **metrics,
    }
    # Convert any numpy values before emitting portable JSON.
    result = json.loads(json.dumps(result, default=lambda value: value.item()))
    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    print(output_text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
