import argparse
import importlib.util
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pygame
import torch

try:
    from .world_model_vae import (
        DEFAULT_MEMORY_FRAMES,
        DEFAULT_MEMORY_STRIDE,
        MODEL_FRAME_SIZE,
        PROJECT_ROOT,
        WORLD_MODEL_WEIGHTS,
        WorldModel,
    )
except ImportError:
    from world_model_vae import (
        DEFAULT_MEMORY_FRAMES,
        DEFAULT_MEMORY_STRIDE,
        MODEL_FRAME_SIZE,
        PROJECT_ROOT,
        WORLD_MODEL_WEIGHTS,
        WorldModel,
    )


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CANVAS_SIZE = 512
SIDE_PANEL_WIDTH = 380
WINDOW_WIDTH = CANVAS_SIZE + SIDE_PANEL_WIDTH
WINDOW_HEIGHT = 640
BG_COLOR = (246, 246, 240)
PANEL_COLOR = (231, 223, 207)
TEXT_COLOR = (44, 44, 44)
ACCENT_COLOR = (201, 106, 61)
BUTTON_COLOR = (252, 248, 241)
BUTTON_BORDER = (150, 112, 79)


def _load_scenario1_module():
    scenario_path = os.path.join(PROJECT_ROOT, "dataset", "scenario1", "scenario1.py")
    if not os.path.exists(scenario_path):
        return None

    spec = importlib.util.spec_from_file_location("scenario1_module", scenario_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_empty_scenario_frame() -> np.ndarray:
    try:
        import pymunk
        import pymunk.pygame_util
    except ImportError as exc:
        raise RuntimeError(
            "pymunk e necessario para gerar o frame inicial identico ao dataset."
        ) from exc

    scenario1 = _load_scenario1_module()
    if scenario1 is None or not hasattr(scenario1, "create_scenario"):
        raise RuntimeError("Nao foi possivel carregar create_scenario de dataset/scenario1/scenario1.py.")

    surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    space = pymunk.Space()
    space.gravity = 0, 980
    draw_options = pymunk.pygame_util.DrawOptions(surface)

    scenario1.create_scenario(space)
    space.step(1 / 60.0)
    surface.fill((255, 255, 255))
    space.debug_draw(draw_options)

    rgb_array = pygame.surfarray.array3d(surface)
    frame_rgb = rgb_array.swapaxes(0, 1)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(frame_bgr, (MODEL_FRAME_SIZE, MODEL_FRAME_SIZE))
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return grayscale.astype(np.uint8)


def action_to_label(action: Optional[dict]) -> str:
    if action is None:
        return "sem acao"
    pos = action.get("pos", [0, 0])
    return f"mouse_down ball ({int(pos[0])}, {int(pos[1])})"


class Button:
    def __init__(self, rect: pygame.Rect, label: str) -> None:
        self.rect = rect
        self.label = label

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        pygame.draw.rect(surface, BUTTON_COLOR, self.rect, border_radius=12)
        pygame.draw.rect(surface, BUTTON_BORDER, self.rect, width=2, border_radius=12)
        text = font.render(self.label, True, TEXT_COLOR)
        surface.blit(text, text.get_rect(center=self.rect.center))

    def contains(self, pos: Tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)


class InteractiveWorldModelSimulator:
    def __init__(self, world_model: WorldModel) -> None:
        pygame.init()
        pygame.display.set_caption("World Model 2D Simulator")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 20)
        self.small_font = pygame.font.SysFont("arial", 16)
        self.world_model = world_model

        self.canvas_rect = pygame.Rect(0, 0, CANVAS_SIZE, CANVAS_SIZE)
        panel_x = CANVAS_SIZE + 24
        button_width = SIDE_PANEL_WIDTH - 48
        button_height = 44
        button_gap = 12
        button_start_y = 252
        self.buttons: Dict[str, Button] = {
            "advance": Button(
                pygame.Rect(panel_x, button_start_y, button_width, button_height),
                "Passar frame",
            ),
            "back": Button(
                pygame.Rect(panel_x, button_start_y + (button_height + button_gap), button_width, button_height),
                "Voltar frame",
            ),
            "auto_play": Button(
                pygame.Rect(panel_x, button_start_y + 2 * (button_height + button_gap), button_width, button_height),
                "Iniciar automatico",
            ),
            "clear_action": Button(
                pygame.Rect(panel_x, button_start_y + 3 * (button_height + button_gap), button_width, button_height),
                "Limpar acao",
            ),
            "reset": Button(
                pygame.Rect(panel_x, button_start_y + 4 * (button_height + button_gap), button_width, button_height),
                "Resetar simulacao",
            ),
        }

        self.reset()

    def reset(self) -> None:
        self.frames: List[np.ndarray] = [create_empty_scenario_frame()]
        self.actions: List[Optional[dict]] = []
        self.fused_history: List[torch.Tensor] = []
        self.current_index = 0
        self.pending_action: Optional[dict] = None
        self.last_result = None
        self.auto_play_enabled = False
        self.auto_play_interval_ms = 180
        self.last_auto_advance_ms = pygame.time.get_ticks()
        self.buttons["auto_play"].label = "Iniciar automatico"

    def create_action_from_canvas(self, mouse_pos: Tuple[int, int]) -> dict:
        x, y = mouse_pos
        x_norm = min(max(x / self.canvas_rect.width, 0.0), 1.0)
        y_norm = min(max(y / self.canvas_rect.height, 0.0), 1.0)
        return {
            "type": "mouse_down",
            "object": "ball",
            "pos": [int(x_norm * SCREEN_WIDTH), int(y_norm * SCREEN_HEIGHT)],
        }

    def draw_multiline(self, text: str, pos: Tuple[int, int], width: int) -> None:
        words = text.split()
        lines: List[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if self.small_font.size(candidate)[0] <= width:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)

        x, y = pos
        for line in lines:
            rendered = self.small_font.render(line, True, TEXT_COLOR)
            self.screen.blit(rendered, (x, y))
            y += 20

    def advance(self) -> None:
        current_frame = self.frames[self.current_index]
        if self.current_index < len(self.frames) - 1:
            self.frames = self.frames[: self.current_index + 1]
            self.actions = self.actions[: self.current_index]
            self.fused_history = self.fused_history[: self.current_index]

        result, fused = self.world_model.predict(
            current_frame,
            self.pending_action,
            self.fused_history,
        )
        self.frames.append(result.frame)
        self.actions.append(self.pending_action)
        self.fused_history.append(fused.cpu())
        self.current_index += 1
        self.last_result = result
        self.pending_action = None

    def go_back(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
        self.set_auto_play(False)

    def set_auto_play(self, enabled: bool) -> None:
        self.auto_play_enabled = enabled
        self.last_auto_advance_ms = pygame.time.get_ticks()
        self.buttons["auto_play"].label = (
            "Pausar automatico" if self.auto_play_enabled else "Iniciar automatico"
        )

    def toggle_auto_play(self) -> None:
        self.set_auto_play(not self.auto_play_enabled)

    def handle_click(self, pos: Tuple[int, int]) -> None:
        if self.canvas_rect.collidepoint(pos):
            self.pending_action = self.create_action_from_canvas(pos)
            return

        for name, button in self.buttons.items():
            if not button.contains(pos):
                continue
            if name == "advance":
                self.advance()
            elif name == "back":
                self.go_back()
            elif name == "auto_play":
                self.toggle_auto_play()
            elif name == "clear_action":
                self.pending_action = None
            elif name == "reset":
                self.reset()
            return

    def draw_frame(self) -> None:
        frame = self.frames[self.current_index]
        rgb_frame = np.repeat(frame[:, :, None], 3, axis=2)
        surface = pygame.surfarray.make_surface(np.transpose(rgb_frame, (1, 0, 2)))
        surface = pygame.transform.scale(surface, self.canvas_rect.size)
        self.screen.blit(surface, (0, 0))
        pygame.draw.rect(self.screen, ACCENT_COLOR, self.canvas_rect, width=4)

        if self.pending_action is not None:
            px = int(self.pending_action["pos"][0] / SCREEN_WIDTH * self.canvas_rect.width)
            py = int(self.pending_action["pos"][1] / SCREEN_HEIGHT * self.canvas_rect.height)
            pygame.draw.circle(self.screen, ACCENT_COLOR, (px, py), 8)
            pygame.draw.circle(self.screen, (255, 240, 230), (px, py), 14, width=2)

    def draw_panel(self) -> None:
        pygame.draw.rect(self.screen, PANEL_COLOR, pygame.Rect(CANVAS_SIZE, 0, SIDE_PANEL_WIDTH, WINDOW_HEIGHT))
        title = self.font.render("Simulador do World Model", True, TEXT_COLOR)
        self.screen.blit(title, (CANVAS_SIZE + 24, 24))

        subtitle = "Clique no canvas para criar uma acao. Depois use Passar frame para pedir a previsao do proximo quadro."
        self.draw_multiline(subtitle, (CANVAS_SIZE + 24, 62), SIDE_PANEL_WIDTH - 48)

        info_lines = [
            f"Frame atual: {self.current_index}",
            f"Frames no historico: {len(self.frames)}",
            f"Acao pendente: {action_to_label(self.pending_action)}",
            f"Automatico: {'ativo' if self.auto_play_enabled else 'pausado'}",
        ]

        y = 132
        for line in info_lines:
            rendered = self.small_font.render(line, True, TEXT_COLOR)
            self.screen.blit(rendered, (CANVAS_SIZE + 24, y))
            y += 24

        for button in self.buttons.values():
            button.draw(self.screen, self.small_font)

        if self.last_result is not None:
            model_text = f"Predicao: {self.last_result.model_status}"
            self.draw_multiline(model_text, (CANVAS_SIZE + 24, 548), SIDE_PANEL_WIDTH - 48)

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_SPACE, pygame.K_RIGHT):
                        self.advance()
                    elif event.key in (pygame.K_BACKSPACE, pygame.K_LEFT):
                        self.go_back()
                    elif event.key == pygame.K_p:
                        self.toggle_auto_play()
                    elif event.key == pygame.K_c:
                        self.pending_action = None
                    elif event.key == pygame.K_r:
                        self.reset()

            now_ms = pygame.time.get_ticks()
            if self.auto_play_enabled and now_ms - self.last_auto_advance_ms >= self.auto_play_interval_ms:
                self.advance()
                self.last_auto_advance_ms = now_ms

            self.screen.fill(BG_COLOR)
            self.draw_frame()
            self.draw_panel()
            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulador interativo para o world model 2D.")
    parser.add_argument(
        "--world-model-weights",
        default=WORLD_MODEL_WEIGHTS,
        help="Checkpoint do preditor de latente do world model.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Dispositivo da inferencia.",
    )
    parser.add_argument(
        "--memory-frames",
        type=int,
        default=DEFAULT_MEMORY_FRAMES,
        help="Quantidade de frames fundidos usados pela memoria temporal.",
    )
    parser.add_argument(
        "--memory-stride",
        type=int,
        default=DEFAULT_MEMORY_STRIDE,
        help="Intervalo entre frames selecionados para a memoria temporal.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    world_model = WorldModel(
        world_model_weights=args.world_model_weights,
        device=args.device,
        memory_frames=args.memory_frames,
        memory_stride=args.memory_stride,
    )
    simulator = InteractiveWorldModelSimulator(world_model)
    simulator.run()


if __name__ == "__main__":
    main()
