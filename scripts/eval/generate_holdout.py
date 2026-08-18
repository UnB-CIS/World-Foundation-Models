"""Gera um conjunto de teste inedito rodando a simulacao do cenario 1.

Por que isto e necessario: os checkpoints do world model foram treinados a partir
dos episodios em `scripts/dataset/`, mas o manifesto de treino/validacao nao foi
versionado. Qualquer metrica medida naqueles videos e potencialmente in-sample.
Aqui geramos episodios NOVOS, com sementes novas, usando exatamente o mesmo motor
de fisica (`scripts/dataset/scenario_1.py`) e o mesmo pipeline de gravacao — o
que garante um conjunto de teste sem contaminacao.

Diferencas de `batch_runner.py`: a renderizacao e offscreen (sem janela) e sem
`clock.tick`, entao a geracao roda muito mais rapido que tempo real.

Uso:
    python scripts/eval/generate_holdout.py --episodes 30
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random

import cv2
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCENARIO_PATH = os.path.join(PROJECT_ROOT, "scripts", "dataset", "scenario_1.py")
HOLDOUT_ROOT = os.path.join(PROJECT_ROOT, "data", "holdout")

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
GRAVITY = 980
FLOOR_COLOR = (211, 211, 211)


def _load_scenario():
    spec = importlib.util.spec_from_file_location("scenario_1_module", SCENARIO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nao foi possivel carregar {SCENARIO_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_actions(rng: random.Random, max_actions: int = 15) -> list[dict]:
    """Mesma distribuicao de `scripts/dataset/generate_random_inputs.py`."""
    actions: list[dict] = []
    current_time = 0.0
    for _ in range(rng.randint(8, max_actions)):
        current_time += rng.uniform(0.2, 0.8)
        if current_time > 10.0:
            break
        actions.append(
            {
                "time": round(current_time, 2),
                "type": "mouse_down",
                "object": "ball",
                "pos": [rng.randint(50, 750), rng.randint(50, 520)],
            }
        )
    return actions


def simulate(actions: list[dict], video_path: str) -> int:
    """Roda a simulacao offscreen e grava o mp4. Retorna o numero de frames.

    A ordem das operacoes (injetar acao -> step -> desenhar) e identica a de
    `run_automated_simulation`, entao os frames tem a mesma semantica temporal
    dos videos de treino.
    """
    scenario = _load_scenario()
    surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    draw_options = pymunk.pygame_util.DrawOptions(surface)

    space = pymunk.Space()
    space.gravity = 0, GRAVITY
    scenario.create_scenario(space)

    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (SCREEN_WIDTH, SCREEN_HEIGHT),
    )
    simulation_time = 0.0
    action_index = 0
    end_time = actions[-1]["time"] + 2.0 if actions else 2.0
    frames = 0

    while simulation_time < end_time:
        dt = 1 / FPS
        while (
            action_index < len(actions)
            and actions[action_index]["time"] <= round(simulation_time, 4)
        ):
            action = actions[action_index]
            if action["type"] == "mouse_down" and action["object"] == "ball":
                scenario.add_ball_at_mouse_position(space, tuple(action["pos"]))
            action_index += 1

        space.step(dt)
        simulation_time += dt

        surface.fill((255, 255, 255))
        pygame.draw.rect(surface, FLOOR_COLOR, (0, 550, SCREEN_WIDTH, 50))
        space.debug_draw(draw_options)

        image = pygame.surfarray.array3d(surface).swapaxes(0, 1)
        writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        frames += 1

    writer.release()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output", default=HOLDOUT_ROOT)
    args = parser.parse_args()

    pygame.init()
    videos_dir = os.path.join(args.output, "videos")
    inputs_dir = os.path.join(args.output, "inputs")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(inputs_dir, exist_ok=True)

    rng = random.Random(args.seed)
    total_frames = 0
    for index in range(1, args.episodes + 1):
        name = f"holdout_{args.seed}_{index:03d}"
        actions = generate_actions(rng)
        with open(os.path.join(inputs_dir, f"{name}.json"), "w") as handle:
            json.dump(actions, handle, indent=4)
        frames = simulate(actions, os.path.join(videos_dir, f"{name}_auto.mp4"))
        total_frames += frames
        print(f"[{index}/{args.episodes}] {name}: {len(actions)} acoes, {frames} frames")

    pygame.quit()
    print(f"\n{args.episodes} episodios ineditos em {args.output} ({total_frames} frames)")


if __name__ == "__main__":
    main()
