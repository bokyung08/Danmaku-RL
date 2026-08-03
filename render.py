import numpy as np
import pygame

import config
from game import Game


def _to_grayscale(rgb):
    """PIL의 Image.convert("L")과 동일한 흑백 값을 계산한다."""
    r, g, b = rgb
    return (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16


class Renderer:
    def __init__(self, render_mode="rgb_array"):
        self.render_mode = render_mode

        pygame.font.init()
        self.font = pygame.font.SysFont(None, config.HUD_FONT_SIZE)

        if render_mode == "human":
            pygame.display.init()
            self.canvas = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            pygame.display.set_caption("Danmaku render test - Arrow keys")
        else:
            self.canvas = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))

        # 학습용 흑백 이미지 전용 캔버스. 색을 칠한 뒤 흑백으로 변환하는 대신
        # 처음부터 흑백 값으로 그려서 변환 과정을 생략한다 (get_grayscale_image 참고).
        self.gray_canvas = pygame.Surface(
            (config.SCREEN_WIDTH, config.SCREEN_HEIGHT), depth=8
        )
        self.gray_canvas.set_palette([(i, i, i) for i in range(256)])
        self.gray_background = _to_grayscale(config.BACKGROUND_COLOR)  # == 16
        self.gray_agent = _to_grayscale(config.AGENT_COLOR)
        self.gray_ball = _to_grayscale(config.BALL_COLOR)


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

    def get_grayscale_image(self, game: Game):
        """agent/ball만 그린 (height, width) 흑백 이미지를 반환한다.

        RGB로 그린 뒤 흑백으로 변환하는 대신 처음부터 흑백 값으로 그려서
        훨씬 빠르다 (RGB 렌더 5.3ms -> 흑백 렌더 0.4ms). 학습 관측 전용 경로라
        get_image()와 달리 배경을 0으로 둔다 (위 gray_background 설명 참고).
        """
        state = game.state
        canvas = self.gray_canvas

        canvas.fill(self.gray_background)
        # agent를 먼저, ball을 나중에 그린다 (draw_canvas와 같은 순서).
        # 둘이 겹칠 때(=충돌 직전) ball이 보이도록 순서를 맞춰야 한다.
        pygame.draw.circle(
            canvas, self.gray_agent,
            (round(state.agent.x), round(state.agent.y)), state.agent.r,
        )
        for ball in state.balls:
            pygame.draw.circle(
                canvas, self.gray_ball, (round(ball.x), round(ball.y)), ball.r,
            )

        pixels = pygame.surfarray.pixels2d(canvas)  # (width, height) 복사 없는 뷰
        image = np.asarray(pixels, dtype=np.uint8).T  # -> (height, width)
        del pixels  # 캔버스 잠금 해제
        return image

    def draw(self, game: Game, view_score=True):
        if self.render_mode != "human": return 

        self.draw_canvas(game, view_score)
        pygame.display.flip()

    def close(self):
        if self.render_mode == "human":
            pygame.display.quit()  # pygame은 살리고 창만 닫음

