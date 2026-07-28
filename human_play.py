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


def check_restart(event):
    return event.type == pygame.KEYDOWN and event.key == pygame.K_r


def check_quit(event):
    return event.type == pygame.QUIT


def play_loop(game, renderer, clock):
    running = True
    while running:
        action = move(*get_key())

        for event in pygame.event.get():
            if check_quit(event):
                running = False
            if game.state.phase == config.PHASE_GAMEOVER and check_restart(event):
                game.reset()

        if game.state.phase == config.PHASE_PLAYING:
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
