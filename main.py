"""사람 플레이 루프 진입점. 키 입력 처리를 전담한다."""
import pygame

import config
from game import Game
from render import Renderer

KEY_ACTION_MAP = {
    pygame.K_UP: config.ACTION_UP,
    pygame.K_DOWN: config.ACTION_DOWN,
    pygame.K_LEFT: config.ACTION_LEFT,
    pygame.K_RIGHT: config.ACTION_RIGHT,
}


def read_input():
    keys = pygame.key.get_pressed()
    for key, action in KEY_ACTION_MAP.items():
        if keys[key]:
            return action
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
