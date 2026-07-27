"""레벨별 공 스폰 테이블 조회 및 레벨 전환 판정."""
import config


def get_spawns(level):
    """레벨에 해당하는 공 스폰 리스트를 반환한다.

    정의된 최고 레벨을 넘으면 마지막 레벨 상태를 유지한다.
    """
    clamped = min(level, config.MAX_LEVEL)
    return list(config.LEVEL_SPAWNS[clamped])


def next_level(level, steps):
    """생존 스텝 수가 기준 이상이면 다음 레벨 번호를 반환한다."""
    threshold = config.LEVEL_UP_STEPS.get(level)
    if threshold is not None and steps >= threshold:
        return level + 1
    return level
