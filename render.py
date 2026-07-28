"""pygame 렌더링 전담. 상태를 그리기만 하고 변경하지 않는다."""
import pygame

import config

AGENT_COLOR = (80, 200, 255)
BALL_COLOR = (255, 90, 90)
BACKGROUND_COLOR = (15, 15, 25)
HUD_COLOR = (230, 230, 230)
GAMEOVER_COLOR = (255, 255, 255)


class Renderer:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        pygame.display.set_caption("탄막 회피 시뮬레이터")
        self.font = pygame.font.SysFont(config.HUD_FONT_NAME, config.HUD_FONT_SIZE)

    def draw(self, game):
        state = game.state
        self.screen.fill(BACKGROUND_COLOR)

        pygame.draw.circle(
            self.screen, AGENT_COLOR,
            (int(state.agent.x), int(state.agent.y)), state.agent.radius,
        )
        for ball in state.balls:
            pygame.draw.circle(
                self.screen, BALL_COLOR,
                (int(ball.x), int(ball.y)), ball.r,
            )

        hud = self.font.render(
            f"생존 스텝: {state.steps}",
            True, HUD_COLOR,
        )
        self.screen.blit(hud, config.HUD_MARGIN)

        pygame.display.flip()
