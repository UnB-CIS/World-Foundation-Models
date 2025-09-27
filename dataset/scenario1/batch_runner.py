import os
import json
from scenario1 import run_automated_simulation 

def run_batch_simulations():
    """
    Varre a pasta 'inputs' e executa uma simulação automatizada para cada arquivo JSON encontrado
    """
    
    inputs_dir = os.path.join(os.path.dirname(__file__), 'inputs')
    
    if not os.path.exists(inputs_dir):
        print(f"Erro: Pasta de inputs não encontrada em {inputs_dir}.")
        return

    json_files = [f for f in os.listdir(inputs_dir) if f.endswith('.json')]
    
    if not json_files:
        print("Nenhum arquivo JSON encontrado na pasta 'inputs'. Execute o gerador de inputs.")
        return
    
    print(f"Encontrados {len(json_files)} arquivos para simular")

    for filename in json_files:
        input_path = os.path.join(inputs_dir, filename)
        
        print(f"\nProcessando input: {filename}")
        
        try:
            with open(input_path, 'r') as f:
                actions = json.load(f)
            
            # Chama a função de simulação, passando as ações e o nome do arquivo.
            # O nome do arquivo (filename) é crucial para nomear o vídeo de saída.
            run_automated_simulation(actions, filename)
            
            print(f"Simulação para {filename} concluída.")
            
        except Exception as e:
            print(f"[ERRO] Falha ao processar {filename}. Motivo: {e}")
            
if __name__ == "__main__":
    run_batch_simulations()