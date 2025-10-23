import os
import json
from multiprocessing import Pool, cpu_count
import time 
from scenario1 import run_automated_simulation 

def process_single_simulation(input_path):
    """
    Função que carrega o JSON e executa a simulação automática.
    Esta função será executada em um processo separado.
    """
    
    filename = os.path.basename(input_path)
    
    try:
        with open(input_path, 'r') as f:
            actions = json.load(f)
        
        # Chama a função de simulação
        run_automated_simulation(actions, filename)
        
        return f"[SUCESSO] Simulação para {filename} concluída."
        
    except Exception as e:
        return f"[ERRO] Falha ao processar {filename}. Motivo: {e}"

def run_batch_simulations_parallel():
    """
    Varre a pasta 'inputs' e executa simulações para todos os JSONs em paralelo.
    """
    start_time = time.time()
    
    inputs_dir = os.path.join(os.path.dirname(__file__), 'inputs')
    
    if not os.path.exists(inputs_dir):
        print(f"Erro: Pasta de inputs não encontrada em {inputs_dir}.")
        return

    json_files = [f for f in os.listdir(inputs_dir) if f.endswith('.json')]

    if not json_files:
        print("Nenhum arquivo JSON encontrado na pasta 'inputs'.")
        return

    # Constroi a lista de caminhos completos
    json_files_paths = [os.path.join(inputs_dir, f) for f in json_files]
    
    num_cores = cpu_count()
    print(f"Encontrados {len(json_files_paths)} arquivos para simular")
    print(f"Utilizando {num_cores} núcleos para processamento paralelo.")

    # Pool de processos: distribui as tarefas entre os núcleos disponíveis
    with Pool(processes=num_cores) as pool:
        # map() aplica a função 'process_single_simulation' a todos os caminhos na lista
        results = pool.map(process_single_simulation, json_files_paths)
    
    end_time = time.time()

    # Exibir Resultados
    print("\n" + "="*50)
    print("RESUMO DO PROCESSAMENTO EM LOTE:")
    for result in results:
        print(result)
    print(f"\nTempo total de processamento: {end_time - start_time:.2f} segundos.")
    print("="*50)

if __name__ == "__main__":
    run_batch_simulations_parallel()