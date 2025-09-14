import pygame
import pymunk
import pymunk.pygame_util
import cv2 
import datetime
import os

def setup_pygame():
    """Inicializa o Pygame e a tela."""
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Cenario 2")
    return screen, pygame.time.Clock()

def add_ball(space, pos):
    """Adiciona uma nova bola no espaco."""
    massa = 1
    raio = 15
    inercia = pymunk.moment_for_circle(massa, 0, raio)
    bola_body = pymunk.Body(massa, inercia)
    bola_body.position = pos
    bola_shape = pymunk.Circle(bola_body, raio)
    bola_shape.elasticity = 0.9
    bola_shape.friction = 0.8
    space.add(bola_body, bola_shape)

def add_box(space, pos):
    """Adiciona uma nova caixa no espaco."""
    massa = 1
    size = 30
    inercia = pymunk.moment_for_box(massa, (size, size))
    caixa_body = pymunk.Body(massa, inercia)
    caixa_body.position = pos
    caixa_shape = pymunk.Poly.create_box(caixa_body, (size, size))
    caixa_shape.elasticity = 0.5
    caixa_shape.friction = 0.7
    space.add(caixa_body, caixa_shape)

def create_static_segment(space, points):
    """Cria uma superficie estatica a partir de uma lista de pontos."""
    if len(points) < 2:
        return
    
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]
        static_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        segment = pymunk.Segment(static_body, p1, p2, 5)
        segment.elasticity = 0.95
        segment.friction = 0.8
        space.add(static_body, segment)

# --- Loop Principal de Simulação e Gravação ---
def run_simulation_and_record():
    """Roda a simulação e grava um vídeo."""
    screen, clock = setup_pygame()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_filename = f"cenario2:{timestamp}.mp4"
    video_path = os.path.join(os.path.dirname(__file__), '..', 'videos', video_filename)
    
    FPS = 60
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(video_path, fourcc, FPS, (800, 600))
    
    space = pymunk.Space()
    space.gravity = 0, 980
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    drawing_mode = False
    drawing_points = []
    current_object_type = 'ball'
    
    running = True
    while running:
        # Gerenciamento de eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if current_object_type == 'ball':
                        add_ball(space, event.pos)
                    elif current_object_type == 'box':
                        add_box(space, event.pos)
                elif event.button == 3:
                    drawing_mode = True
                    drawing_points.append(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    drawing_mode = False
                    if len(drawing_points) > 1:
                        create_static_segment(space, drawing_points)
                    drawing_points = []
            elif event.type == pygame.MOUSEMOTION:
                if drawing_mode:
                    drawing_points.append(event.pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_b:
                    current_object_type = 'ball'
                    print("Modo: Criar Bolas")
                elif event.key == pygame.K_q:
                    current_object_type = 'box'
                    print("Modo: Criar Caixas")

        # Criar varias bolas com Shift + Botao Esquerdo 
        mouse_buttons = pygame.mouse.get_pressed()
        keys = pygame.key.get_mods()
        if mouse_buttons[0] and keys & pygame.KMOD_SHIFT and current_object_type == 'ball':
            add_ball(space, pygame.mouse.get_pos())

        # Limpar a tela
        screen.fill((255, 255, 255))
        
        if drawing_mode and len(drawing_points) > 1:
            pygame.draw.lines(screen, (0, 0, 0), False, drawing_points, 2)

        space.debug_draw(draw_options)
        space.step(1 / 60.0)

        pygame.display.flip()

        img_array = pygame.surfarray.array3d(screen)
        img_array = cv2.cvtColor(img_array.swapaxes(0, 1), cv2.COLOR_RGB2BGR)
        out.write(img_array)

        clock.tick(60)

    out.release()
    pygame.quit()
    print(f"Vídeo salvo como {video_path}")

if __name__ == "__main__":
    run_simulation_and_record()