import argparse
import os
from PS import run_parameter_server
import torch
import torch.multiprocessing as mp
from Utils import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parameter-Server RPC based training")
    parser.add_argument(
        "--ws",
        type=int,
        default=None,
        help="World size.")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Global rank of this process. Pass in 0 for master.")
    parser.add_argument(
        "--psip",
        type=str,
        default=None,
        help="The IP of master (PS)")
    parser.add_argument(
        "--psport",
        type=str,
        default=None,
        help="The port of master (PS)")
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="The algorithm to test")
    
    args = parser.parse_args()
    assert args.ws is not None, "must provide world size."
    assert args.rank is not None, "must provide rank argument."
    assert args.psip is not None, "must provide ip of master."
    assert args.psport is not None, "must provide port of master."
    assert args.algorithm is not None, "must provide the algorithm to run."

    os.environ['MASTER_ADDR'] = args.psip
    os.environ["MASTER_PORT"] = args.psport
    processes = []
    world_size = args.ws

    if args.rank == 0:
        p = mp.Process(target=run_parameter_server, args=(0, world_size,)) # run_parameter_server defined in PS.py
        p.start()
        processes.append(p)
    else:
        if args.algorithm == 'localSGD':
            from Workers.localSGDWorker import run_worker
        elif args.algorithm == 'OSP':
            from Workers.OSPWorker import run_worker
        elif args.algorithm == 'LOSP':
            from Workers.LOSPWorker import run_worker

        p = mp.Process(
            target=run_worker, # run_worker defined in Worker.py
            args=(
                args.rank,
                world_size,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()