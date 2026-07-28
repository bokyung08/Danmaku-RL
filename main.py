"""사람 플레이 루프 진입점. 키 입력 처리를 전담한다."""
import pygame

import config
from game import Game
from render import Renderer

def read_input():
    keys = pygame.key.get_pressed()
    up = keys[pygame.K_UP]
    down = keys[pygame.K_DOWN]
    left = keys[pygame.K_LEFT]
    right = keys[pygame.K_RIGHT]

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


def is_restart_key(event):
    return event.type == pygame.KEYDOWN and event.key == pygame.K_r


def main():
    game = Game()
    renderer = Renderer()
    clock = pygame.time.Clock()

    running = True
    while running:
        action = read_input()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if game.state.phase == config.PHASE_GAMEOVER and is_restart_key(event):
                game.reset()

        if game.state.phase == config.PHASE_PLAYING:
            game.step(action)

        renderer.draw(game)
        clock.tick(config.FPS)

    pygame.quit()
    print("게임을 종료합니다.")


if __name__ == "__main__":
    main()
