import config
from game import Game


def test_reset_sets_initial_state():
    game = Game()
    state = game.state
    assert state.agent.x == config.SCREEN_WIDTH / 2
    assert state.agent.y == config.SCREEN_HEIGHT / 2
    assert state.level == 1
    assert state.steps == 0
    assert state.alive is True
    assert state.phase == config.PHASE_PLAYING
    assert len(state.balls) == len(config.LEVEL_SPAWNS[1])


def test_determinism_same_actions_same_result():
    actions = [config.ACTION_RIGHT, config.ACTION_UP, config.ACTION_STOP, config.ACTION_DOWN] * 50

    game_a = Game()
    for action in actions:
        game_a.step(action)

    game_b = Game()
    for action in actions:
        game_b.step(action)

    assert game_a.state.agent.x == game_b.state.agent.x
    assert game_a.state.agent.y == game_b.state.agent.y
    assert game_a.state.steps == game_b.state.steps
    assert [(b.x, b.y, b.vx, b.vy) for b in game_a.state.balls] == \
        [(b.x, b.y, b.vx, b.vy) for b in game_b.state.balls]


def test_agent_clipped_to_screen_bounds():
    game = Game()
    for _ in range(200):
        game.step(config.ACTION_LEFT)
    assert game.state.agent.x == game.state.agent.radius

    game.reset()
    for _ in range(200):
        game.step(config.ACTION_UP)
    assert game.state.agent.y == game.state.agent.radius

    game.reset()
    for _ in range(200):
        game.step(config.ACTION_RIGHT)
    assert game.state.agent.x == config.SCREEN_WIDTH - game.state.agent.radius

    game.reset()
    for _ in range(200):
        game.step(config.ACTION_DOWN)
    assert game.state.agent.y == config.SCREEN_HEIGHT - game.state.agent.radius


def test_wall_reflection_keeps_ball_inside_bounds():
    game = Game()
    prev_velocities = [(ball.vx, ball.vy) for ball in game.state.balls]
    consecutive_flip_counts = [{"vx": 0, "vy": 0} for _ in game.state.balls]

    for _ in range(300):
        game.step(config.ACTION_STOP)
        for i, ball in enumerate(game.state.balls):
            assert ball.radius <= ball.x <= config.SCREEN_WIDTH - ball.radius
            assert ball.radius <= ball.y <= config.SCREEN_HEIGHT - ball.radius

            prev_vx, prev_vy = prev_velocities[i]
            counts = consecutive_flip_counts[i]

            vx_flipped = (ball.vx * prev_vx) < 0
            vy_flipped = (ball.vy * prev_vy) < 0

            counts["vx"] = counts["vx"] + 1 if vx_flipped else 0
            counts["vy"] = counts["vy"] + 1 if vy_flipped else 0

            assert counts["vx"] < 2, "vx flipped sign on two consecutive steps (jitter)"
            assert counts["vy"] < 2, "vy flipped sign on two consecutive steps (jitter)"

            prev_velocities[i] = (ball.vx, ball.vy)


def test_collision_ends_game_immediately():
    game = Game()
    ball = game.state.balls[0]
    ball.vx = 0
    ball.vy = 0
    game.state.agent.x = ball.x
    game.state.agent.y = ball.y

    game.step(config.ACTION_STOP)

    assert game.state.alive is False
    assert game.state.phase == config.PHASE_GAMEOVER


def test_reset_after_gameover_restores_full_state():
    game = Game()
    ball = game.state.balls[0]
    ball.vx = 0
    ball.vy = 0
    game.state.agent.x = ball.x
    game.state.agent.y = ball.y

    game.step(config.ACTION_STOP)

    assert game.state.phase == config.PHASE_GAMEOVER

    game.reset()

    assert game.state.phase == config.PHASE_PLAYING
    assert game.state.steps == 0
    assert game.state.level == 1
    assert game.state.alive is True
    assert len(game.state.balls) == len(config.LEVEL_SPAWNS[1])


def test_step_ignored_after_gameover():
    game = Game()
    game.state.phase = config.PHASE_GAMEOVER
    steps_before = game.state.steps

    game.step(config.ACTION_RIGHT)

    assert game.state.steps == steps_before


def test_level_transitions_at_exact_step_threshold():
    game = Game()
    for _ in range(config.LEVEL_UP_STEPS[1]):
        game.step(config.ACTION_STOP)

    assert game.state.level == 2
    assert len(game.state.balls) == len(config.LEVEL_SPAWNS[2])


def test_headless_runs_many_steps_without_render():
    game = Game()
    for _ in range(2000):
        if game.state.phase != config.PHASE_PLAYING:
            break
        game.step(config.ACTION_STOP)
    assert game.state.steps > 0
    assert game.state.level >= 1
