import pygame

import config
from game import Game

class Renderer:
    def __init__(self, render_mode="rgb_array"):
        self.render_mode = render_mode

        self.canvas = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        
        pygame.font.init()
        self.font = pygame.font.SysFont(
            " ", config.HUD_FONT_SIZE
        )
        
        self.screen = None
        if render_mode == "human":
            pygame.display.init()
            self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            pygame.display.set_caption("Danmaku render test - Arrow keys")


    def draw_canvas(self, game: Game, view_score=True):
        state = game.state
        agent, balls, score = state.agent, state.balls, state.score

        self.canvas.fill(config.BACKGROUND_COLOR)
        #1 agent를 circle로 표시
        pygame.draw.circle(
            self.canvas,
            config.AGENT_COLOR,
            (round(agent.x), round(agent.y)),
            agent.r
        )
        #2 ball들을 circle로 표시
        for ball in balls:
            pygame.draw.circle(
                self.canvas,
                config.BALL_COLOR,
                (round(ball.x), round(ball.y)),
                ball.r
            )

        if view_score:
            #3 score는 왼쪽 위에 표시
            text = self.font.render(f"Score : {score}", True, config.HUD_COLOR)
            self.canvas.blit(text, config.HUD_MARGIN)

    def get_image(self, game:Game, view_score=False):
        self.draw_canvas(game, view_score)

        image = pygame.surfarray.array3d(self.canvas)  # (width, height, channel)
        return image.transpose(1,0,2)  # (height, width, channel)

    def draw(self, game: Game, view_score=True):
        self.draw_canvas(game, view_score)
        self.screen.blit(self.canvas, (0,0))
        pygame.display.flip()
