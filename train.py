import torch
import pickle
import argparse
import numpy as np

from pathlib import Path
from game import State
from dual_network import DualNetwork
from self_play import self_play


def load_data():
    history_path = sorted(Path('./data').glob('*.history'))[-1]

    with history_path.open(mode='rb') as f:
        return pickle.load(f)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\nDevice : ', device)

    net = DualNetwork(num_residual_block=args.num_residual_block, num_filters=args.num_filters)
    for _ in range(args.cycle):

        print('\n<Load Dataset>')
        history = load_data()
        xs, y_policies, y_values = zip(*history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reinforcement Train")

    parser.add_argument('--model_path', type=str, default='./model/best.pth')
    parser.add_argument('--pv_eval_count', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--num_residual_block', type=int, default=16)
    parser.add_argument('--num_filters', type=int, default=128)
    parser.add_argument('--self_count', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--cycle', type=int, default=10)