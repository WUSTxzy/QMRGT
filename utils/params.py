import argparse
import random

import numpy as np
import torch



def parse_args():
    parser = argparse.ArgumentParser()

    # Data Splits
    parser.add_argument("--train", default='train,valid')
    parser.add_argument("--valid", default='testdev')
    parser.add_argument("--test", default=None)

    # Training Hyper-parameters
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=256)
    parser.add_argument('--optim', default='bert')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=9595, help='random seed')

    # Debugging
    parser.add_argument('--output', type=str, default='results/gqa/QMRGT')
    parser.add_argument("--tqdm", action='store_const', default=True, const=True)

    # Model Loading
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model.')

    # Optimization
    parser.add_argument("--mceLoss", dest='mce_loss', action='store_const', default=False, const=True)
    # parser.add_argument("--bceloss", dest='bce_loss', action='store_const', default=False, const=True)



    # Training configuration
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument("--numWorkers", dest='num_workers', default=0)

    # Parse the arguments.
    args = parser.parse_args()


    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


args = parse_args()
