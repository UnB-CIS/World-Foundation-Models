import json
import random
import os
import datetime


def generate_random_input(num_files=200, max_actions=15, max_time=10.0):
    """Gera multiplos arquivos JSON com ações aleatorias.

    Changes from original:
    - Action interval reduced from 0.3-1.5s to 0.2-0.8s so each simulation
      has more click events, increasing their proportion in the dataset.
    - Y position covers the full screen (50-520) so the model learns to
      respond to clicks anywhere, not just near the floor.
    """

    inputs_dir = os.path.join(os.path.dirname(__file__), 'inputs')
    os.makedirs(inputs_dir, exist_ok=True)

    for i in range(num_files):
        actions = []

        num_actions = random.randint(8, max_actions)  # was 5-15, now 8-15

        current_time = 0.0
        for _ in range(num_actions):
            # Shorter interval — more clicks per simulation
            current_time += random.uniform(0.2, 0.8)  # was 0.3-1.5

            if current_time > max_time:
                break

            pos_x = random.randint(50, 750)
            pos_y = random.randint(50, 520)  # full screen coverage

            action = {
                "time": round(current_time, 2),
                "type": "mouse_down",
                "object": "ball",
                "pos": [pos_x, pos_y],
            }
            actions.append(action)

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = f"random_input_{timestamp}_{i+1:03d}.json"
        file_path = os.path.join(inputs_dir, filename)

        with open(file_path, 'w') as f:
            json.dump(actions, f, indent=4)

        print(f"Gerado: {filename} ({len(actions)} acoes)")


if __name__ == "__main__":
    generate_random_input(num_files=400, max_actions=15)
