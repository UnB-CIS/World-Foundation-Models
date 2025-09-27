import pygame
import pymunk
import pymunk.pygame_util
import cv2 
import datetime
import os
import json 

def setup_pygame():
    """Inicializa o Pygame e a tela."""
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Cenario 1")
    return screen, pygame.time.Clock()

def create_scenario(space):
    """Cria o chão estático do cenário."""
    body_chao = pymunk.Body(body_type=pymunk.Body.STATIC)
    segment_chao = pymunk.Segment(body_chao, (0, 550), (800, 550), 5)
    segment_chao.elasticity = 0.9
    segment_chao.friction = 1.0
    space.add(body_chao, segment_chao)

def add_ball_at_mouse_position(space, pos):
    """Adiciona uma nova bola no espaço, na posição do mouse"""
    massa = 1
    raio = 15
    inercia = pymunk.moment_for_circle(massa, 0, raio)
    bola_body = pymunk.Body(massa, inercia)
    bola_body.position = pos
    bola_shape = pymunk.Circle(bola_body, raio)
    bola_shape.elasticity = 0.9
    bola_shape.friction = 0.8
    space.add(bola_body, bola_shape)

# Loop principal de simulação, gravação e coleta de dados 
def run_simulation_and_record():
    """Roda a simulação e grava um vídeo(mp4) e um arquivo de dados(json)"""
    screen, clock = setup_pygame()

    # Configurar caminhos e nomes de arquivos
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_filename = f"cenario_1_{timestamp}.mp4"
    data_filename = f"cenario_1_data_{timestamp}.json"
    
    # Caminho base: sobe um nivel (do 'scenarios' para o 'dataset')
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    
    video_path = os.path.join(base_dir, 'videos', video_filename)
    data_path = os.path.join(base_dir, 'inputs', data_filename)

    # Garante que a pasta 'inputs' exista
    os.makedirs(os.path.join(base_dir, 'inputs'), exist_ok=True)
    
    # Configurar Video e Pymunk
    FPS = 60
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(video_path, fourcc, FPS, (800, 600))

    space = pymunk.Space()
    space.gravity = 0, 980
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    create_scenario(space)
    
    # Variaveis de Coleta de Dados
    simulation_time = 0.0
    recorded_actions = []
    
    running = True
    while running:
        dt = 1 / 60.0 # Passo de tempo fixo para a simulacao

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    add_ball_at_mouse_position(space, event.pos)
                    
                    # Coleta de dados da açao
                    action = {
                        "time": round(simulation_time, 4),
                        "type": "mouse_down",
                        "object": "ball",
                        "pos": [event.pos[0], event.pos[1]]
                    }
                    recorded_actions.append(action)

        # Atualizar Simulacao e Tempo
        space.step(dt)
        simulation_time += dt

        # Limpar a tela e desenhar
        screen.fill((255, 255, 255))
        space.debug_draw(draw_options)
        pygame.display.flip()

        # Gravar Frame
        img_array = pygame.surfarray.array3d(screen)
        img_array = cv2.cvtColor(img_array.swapaxes(0, 1), cv2.COLOR_RGB2BGR)
        out.write(img_array)

        clock.tick(60)

    # Salvar o arquivo JSON
    with open(data_path, 'w') as f:
        json.dump(recorded_actions, f, indent=4)
    
    # Liberar recursos
    out.release()
    pygame.quit()
    print(f"\nVídeo salvo: {video_path}")
    print(f"Dados de Input (JSON) salvos: {data_path}")

if __name__ == "__main__":
    run_simulation_and_record()