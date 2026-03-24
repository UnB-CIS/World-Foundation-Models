import json
import random
import os
import datetime

def generate_random_input(num_files=10, max_actions=15, max_time=10.0):
    """Gera multiplos arquivos JSON com ações aleatorias."""
    
    inputs_dir = os.path.join(os.path.dirname(__file__), 'inputs')
    os.makedirs(inputs_dir, exist_ok=True) # Checa se a pasta 'inputs' já existe
    
    for i in range(num_files):
        actions = []
        
        # 1. Gera o numero de acoes para este arquivo (entre 5 e max_actions)
        num_actions = random.randint(5, max_actions)
        
        # 2. Gera as acoes com tempo crescente
        current_time = 0.0
        for _ in range(num_actions):
            # Tempo aleatorio entre 0.3s e 1.5s apos a ultima acao
            current_time += random.uniform(0.3, 1.5)
            
            # Limita o tempo total
            if current_time > max_time:
                break
            
            # Posicao aleatoria dentro da tela (800x600)
            pos_x = random.randint(50, 750)
            pos_y = random.randint(50, 500)
            
            action = {
                "time": round(current_time, 2),
                "type": "mouse_down",
                "object": "ball", # No Cenario 1, so ha bolas
                "pos": [pos_x, pos_y]
            }
            actions.append(action)

        # 3. Salva o arquivo
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"random_input_{timestamp}_{i+1:02d}.json"
        file_path = os.path.join(inputs_dir, filename)
        
        with open(file_path, 'w') as f:
            json.dump(actions, f, indent=4)
            
        print(f"Gerado: {filename} ({len(actions)} acoes)")

if __name__ == "__main__":
    generate_random_input(num_files=200, max_actions=15)