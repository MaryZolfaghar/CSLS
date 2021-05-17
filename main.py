import argparse
import torch 
import pickle
import random
import numpy as np 

from dataset import get_loaders
from models import EpisodicSystem, CorticalSystem
from train import train
from test import test
from analyze import analyze


parser = argparse.ArgumentParser()
# Setup
parser.add_argument('--use_cuda', action='store_true',
                    help='Use GPU, if available')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('--print_every', type=int, default=200,
                    help='Number of steps before printing average loss')
parser.add_argument('--out_file', default='results.P')
# Episodic memory system
parser.add_argument('--N_episodic', type=int, default=1000,
                    help='Number of steps for pre-training episodic system')
parser.add_argument('--bs_episodic', type=int, default=16,
                    help='Minibatch size for episodic system')
parser.add_argument('--lr_episodic', type=float, default=0.001,
                    help='Learning rate for episodic system')
# Cortical system
parser.add_argument('--use_images', action='store_true',
                    help='Use full face images and CNN for cortical system')
parser.add_argument('--image_dir', default='images/',
                    help='Path to directory containing face images')
parser.add_argument('--N_cortical', type=int, default=10000,
                    help='Number of steps for training cortical system')
parser.add_argument('--bs_cortical', type=int, default=32,
                    help='Minibatch size for cortical system')
parser.add_argument('--lr_cortical', type=float, default=0.001,
                    help='Learning rate for cortical system')

def main(args):
    # CUDA
    if args.use_cuda:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
    else:
        use_cuda = False
        device = "cpu"
    args.device = device
    print("Using CUDA: ", use_cuda)

    # Random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Episodic memory system: Pre-train, test, analyze (hub retrieval)
    meta = True # meta-learning for episodic memory system
    episodic_system = EpisodicSystem().to(device)
    loaders = get_loaders(batch_size=args.bs_episodic, meta=meta, 
                          use_images=False, image_dir=args.image_dir, 
                          n_episodes=args.N_episodic)
    train_loader, test_loader = loaders
    episodic_train_results = train(meta, episodic_system, train_loader, args)
    episodic_test_results = test(meta, episodic_system, test_loader, args)
    episodic_analysis = analyze(meta, episodic_system, test_loader, args)
    episodic_results = {'train' : episodic_train_results,
                        'test' : episodic_test_results,
                        'analysis' : episodic_analysis}

    # Cortical system: Train, test, analyze (PCA, correlation)
    meta = False # cortical learning is vanilla
    cortical_system = CorticalSystem(use_images=args.use_images).to(device)
    loaders = get_loaders(batch_size=args.bs_cortical, meta=False,
                          use_images=args.use_images, image_dir=args.image_dir,
                          n_episodes=None)
    train_loader, test_loader = loaders
    cortical_train_results = train(meta, cortical_system, train_loader, args)
    cortical_test_results = test(meta, cortical_system, test_loader, args)
    cortical_analysis = analyze(meta, cortical_system, test_loader, args)
    cortical_results = {'train' : cortical_train_results,
                        'test' : cortical_test_results,
                        'analysis' : cortical_analysis}
    
    # Save results
    results = {'Episodic' : episodic_results,
               'Cortical' : cortical_results}
    with open(args.out_file, 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)