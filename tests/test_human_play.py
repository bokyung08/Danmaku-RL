import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import config
from human_play import move


def test_move_returns_stop_when_no_keys_pressed():
    assert move(False, False, False, False) == config.ACTION_STOP


def test_move_returns_cardinal_actions():
    assert move(True, False, False, False) == config.ACTION_UP
    assert move(False, True, False, False) == config.ACTION_DOWN
    assert move(False, False, True, False) == config.ACTION_LEFT
    assert move(False, False, False, True) == config.ACTION_RIGHT


def test_move_returns_diagonal_actions():
    assert move(True, False, True, False) == config.ACTION_UP_LEFT
    assert move(True, False, False, True) == config.ACTION_UP_RIGHT
    assert move(False, True, True, False) == config.ACTION_DOWN_LEFT
    assert move(False, True, False, True) == config.ACTION_DOWN_RIGHT
