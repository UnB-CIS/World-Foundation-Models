import os
import json
import time
from scripts.dataset.scenario_1 import run_automated_simulation


def process_single_simulation(input_path):
    """
    Carrega o JSON e executa a simulação automática.

    Runs serially (not in a multiprocessing Pool) to avoid corrupted MP4 files.
    When multiple cv2.VideoWriter processes run simultaneously via multiprocessing,
    the moov atom is often never written, producing unreadable video files.
    Serial execution is slower but guarantees every video is valid.
    """
    filename = os.path.basename(input_path)

    try:
        with open(input_path, 'r') as f:
            actions = json.load(f)

        run_automated_simulation(actions, filename)
        return f"[SUCESSO] Simulação para {filename} concluída."

    except Exception as e:
        return f"[ERRO] Falha ao processar {filename}. Motivo: {e}"


def run_batch_simulations():
    """
    Varre a pasta 'inputs' e executa simulações para todos os JSONs em série.
    """
    start_time = time.time()

    inputs_dir = os.path.join(os.path.dirname(__file__), 'inputs')

    if not os.path.exists(inputs_dir):
        print(f"Erro: Pasta de inputs não encontrada em {inputs_dir}.")
        return

    json_files = sorted([f for f in os.listdir(inputs_dir) if f.endswith('.json')])

    if not json_files:
        print("Nenhum arquivo JSON encontrado na pasta 'inputs'.")
        return

    json_files_paths = [os.path.join(inputs_dir, f) for f in json_files]

    print(f"Encontrados {len(json_files_paths)} arquivos para simular")
    print("Executando em série para garantir vídeos válidos (sem multiprocessing).")

    results = []
    for i, path in enumerate(json_files_paths):
        print(
            f"[{i+1}/{len(json_files_paths)}] Processando {os.path.basename(path)}..."
        )
        result = process_single_simulation(path)
        results.append(result)
        if "[ERRO]" in result:
            print(result)

    end_time = time.time()

    successes = sum(1 for r in results if "[SUCESSO]" in r)
    failures = len(results) - successes

    print("\n" + "=" * 50)
    print("RESUMO DO PROCESSAMENTO EM LOTE:")
    print(f"  Sucesso: {successes}/{len(results)}")
    print(f"  Falhas:  {failures}/{len(results)}")
    if failures > 0:
        for r in results:
            if "[ERRO]" in r:
                print(f"  {r}")
    print(f"\nTempo total de processamento: {end_time - start_time:.2f} segundos.")
    print("=" * 50)


if __name__ == "__main__":
    run_batch_simulations()
