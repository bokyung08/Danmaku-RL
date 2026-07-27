import config
import level


def test_get_spawns_returns_level_table():
    assert level.get_spawns(1) == config.LEVEL_SPAWNS[1]
    assert level.get_spawns(3) == config.LEVEL_SPAWNS[3]


def test_get_spawns_clamps_to_max_level():
    assert level.get_spawns(99) == config.LEVEL_SPAWNS[config.MAX_LEVEL]


def test_next_level_stays_below_threshold():
    assert level.next_level(1, config.LEVEL_UP_STEPS[1] - 1) == 1


def test_next_level_advances_at_exact_threshold():
    assert level.next_level(1, config.LEVEL_UP_STEPS[1]) == 2


def test_next_level_stays_at_max_level():
    assert level.next_level(config.MAX_LEVEL, 999999) == config.MAX_LEVEL
