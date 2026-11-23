"""Punto de entrada del pipeline: regenera datos, entrena los cinco modelos o reproduce la comparación de 70 distritos."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modo', choices=['equipo70','experimento-v1','datos'], default='equipo70')
    parser.add_argument('--reutilizar-datos', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    import pandas as pd
    import torch
    from threadpoolctl import threadpool_limits
    import experimento_clima as ex
    import integracion_clima as ic
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.modo in ('datos','equipo70'):
            required = [ROOT/'data/processed/dataset_cajamarca.csv', ROOT/'data/processed/clima_mensual_cajamarca.csv']
            if args.modo == 'datos' or not all(p.exists() for p in required):
                ic.run(ex.RAW, ROOT/'data/raw/clima', ROOT/'data/processed', ROOT/'data/raw/external')
            if args.modo == 'equipo70':
                import comparacion_70_distritos
                comparacion_70_distritos.main()
        else:
            if args.reutilizar_datos:
                datasets = {(p,v): pd.read_csv(ex.DATA/p/f'{v}.csv', dtype={'ubigeo':str},
                             parse_dates=['fecha_mes'], low_memory=False)
                            for p in ex.PROTOCOLS for v in ex.VARIANTS}
            else:
                datasets = ex.build_datasets()
            ex.run_experiments(datasets)
            from reporte_experimento_clima import main as report
            from verificar_experimento_clima import main as audit
            report(); audit()


if __name__ == '__main__':
    main()
