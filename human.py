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
    fixed_dt = 1.0 / config.PHYSICS_FPS # 초당 60번 게임 step을 실행한다고 가정하면 한 번 실행하는 시간은 1/60 , 한 스텝이 진행되는 시간을 고정한다. 
    accumulator = 0.0 # 시간 누적기 

    while True:
        frame_dt = clock.tick(config.RENDER_FPS)/1000.0 # 실제로 흐른 시간 측정(ms) / 1000 
        frame_dt = min(frame_dt, config.MAX_FRAME_TIME) # 정지 등으로 인해 실제로 흐른 시간 값이 커져도 물리 업데이트가 한꺼번에
                                                        # 지나치게 많이 실행되는 것을 방지
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and not game.state.alive:
                    game.reset()
                    accumulator = 0.0 
                elif event.key == pygame.K_ESCAPE:
                    return
                
        action = move(*get_key())
        if game.state.alive: 
            accumulator += frame_dt * config.GAME_SPEED # 실제 흐른 시간 * 게임 속도 
            while accumulator >= fixed_dt: # STEP 한 번을 실행할 만큼 시간이 누적됐는지 확인 
                game.step(action)
                accumulator -= fixed_dt # 한 번 실행했으므로 실행한 시간 차감 
        else: 
            accumulator = 0.0
        renderer.draw(game)

def main():
    game = Game()
    renderer = Renderer()
    clock = pygame.time.Clock()
    
    play_loop(game, renderer, clock)

    pygame.quit()

if __name__ == "__main__": 
    main()
