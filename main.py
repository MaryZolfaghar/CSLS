import argparse
import torch 
import pickle
import random
import numpy as np 

from dataset import get_loaders
from models import EpisodicSystem, CorticalSystem
from train import train
from test import test
from analyze import analyze_episodic, analyze_cortical


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
parser.add_argument('--N_cortical', type=int, default=1000,
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
    data = get_loaders(batch_size=args.bs_episodic, meta=meta, 
                       use_images=False, image_dir=args.image_dir, 
                       n_episodes=args.N_episodic)
    train_data, train_loader, test_data, test_loader = data
    episodic_train_losses = train(meta, episodic_system, train_loader, args)
    episodic_train_acc = test(meta, episodic_system, train_loader, args)
    episodic_test_acc = test(meta, episodic_system, test_loader, args)
    episodic_analysis = analyze_episodic(episodic_system, test_data, args)
    print("Episodic system training accuracy:", episodic_train_acc)
    print("Episodic system testing accuracy:", episodic_test_acc)
    episodic_results = {'loss' : episodic_train_losses,
                        'train_acc': episodic_train_acc,
                        'test_acc' : episodic_test_acc,
                        'analysis' : episodic_analysis}

    # Cortical system: Train, test, analyze (PCA, correlation)
    meta = False # cortical learning is vanilla
    cortical_system = CorticalSystem(use_images=args.use_images).to(device)
    data = get_loaders(batch_size=args.bs_cortical, meta=False,
                       use_images=args.use_images, image_dir=args.image_dir,
                       n_episodes=None)
    train_data, train_loader, test_data, test_loader = data
    cortical_train_losses = train(meta, cortical_system, train_loader, args)
    cortical_train_acc = test(meta, cortical_system, train_loader, args)
    cortical_test_acc = test(meta, cortical_system, test_loader, args)
    cortical_analysis = analyze_cortical(cortical_system, test_data, args)
    print("Cortical system training accuracy:", cortical_train_acc)
    print("Cortical system testing accuracy:", cortical_test_acc)
    cortical_results = {'loss': cortical_train_losses,
                        'train_acc': cortical_train_acc,
                        'test_acc': cortical_test_acc,
                        'analysis': cortical_analysis}
    
    # Save results
    results = {'Episodic' : episodic_results,
               'Cortical' : cortical_results}
    with open(args.out_file, 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)