import os
# Desactivar logs informativos y advertencias de compilación de TensorFlow/CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'

import yaml
from src.pipeline import Pipeline

def main():
    print("Iniciando Orquestador de EMBase...")
    
    # 1. Cargar la configuración
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # 2. Inicializar y correr
    pipeline = Pipeline(config)
    resultados = pipeline.run()
    
    exitosos = [r for r in resultados if r is not None]
    fallidos = len(resultados) - len(exitosos)
    
    print("\n" + "="*50)
    if fallidos == 0 and len(exitosos) > 0:
        print("[OK] EJECUCION FINALIZADA: Todos los metodos corrieron con exito.")
    elif len(exitosos) == 0:
        print("[ERROR] EJECUCION FALLIDA: Ningun metodo pudo terminar correctamente.")
        print("   -> Revisa la consola arriba para leer los '[!] ERROR FATAL'.")
        print("   -> Asegurate de tener activado tu entorno: conda activate pysptools_env")
    else:
        print(f"[WARNING] EJECUCION PARCIAL: {len(exitosos)} metodos tuvieron exito, pero {fallidos} fallaron.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
