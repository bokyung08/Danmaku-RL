import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import config
from game import Game
from render import Renderer


def test_draw_does_not_mutate_state():
    game = Game()
    game.step(config.ACTION_RIGHT)
    renderer = Renderer()

    def snapshot():
        s = game.state
        return (s.agent.x, s.agent.y, s.steps, s.level, s.phase,
                [(b.x, b.y, b.vx, b.vy) for b in s.balls])

    before = snapshot()
    renderer.draw(game)
    renderer.draw(game)
    after = snapshot()

    assert before == after


def test_draw_gameover_screen_does_not_raise():
    game = Game()
    game.state.phase = config.PHASE_GAMEOVER
    renderer = Renderer()

    renderer.draw(game)
