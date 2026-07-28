"""사람 플레이 루프 진입점. 키 입력 처리를 전담한다."""
import pygame

import config
from game import Game
from render import Renderer


def get_key():
    keys = pygame.key.get_pressed()
    return (
        keys[pygame.K_UP],
        keys[pygame.K_DOWN],
        keys[pygame.K_LEFT],
        keys[pygame.K_RIGHT],
    )


def move(up, down, left, right):
    if up and left:
        return config.ACTION_UP_LEFT
    if up and right:
        return config.ACTION_UP_RIGHT
    if down and left:
        return config.ACTION_DOWN_LEFT
    if down and right:
        return config.ACTION_DOWN_RIGHT
    if up:
        return config.ACTION_UP
    if down:
        return config.ACTION_DOWN
    if left:
        return config.ACTION_LEFT
    if right:
        return config.ACTION_RIGHT
    return config.ACTION_STOP


def play_loop(game, renderer, clock):
    running = True
    while running:
        action = move(*get_key())
        game.step(action)

        renderer.draw(game)
        clock.tick(config.FPS)


def main():
    game = Game()
    renderer = Renderer()
    clock = pygame.time.Clock()

    play_loop(game, renderer, clock)

    pygame.quit()
    print("게임을 종료합니다.")


if __name__ == "__main__":
    main()
