import pygame
import config 
from game import Game 
from render import Renderer


def get_key(): 
    keys = pygame.key.get_pressed()
    return( keys[pygame.K_UP], keys[pygame.K_DOWN], keys[pygame.K_LEFT], keys[pygame.K_RIGHT]) # (T, F, F, T)

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
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and not game.state.alive:
                    game.reset()
                elif event.key == pygame.K_ESCAPE:
                    return
                
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

if __name__ == "__main__": 
    main()
