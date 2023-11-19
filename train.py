import os
import torch
import pickle
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from game import State
from dual_network import DualNetwork
from self_play import self_play
from data import TicTacToeDataset

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\nDevice : ', device)
    
    # Dataset & DataLoader
    print('\n<Load Dataset>')
    dataset = TicTacToeDataset()
    data_loader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # Model, Criterion, Optimizer
    net = DualNetwork(num_residual_block=args.num_residual_block, num_filters=args.num_filters).to(device)
    if args.model_path is not None:
        net.load_state_dict(torch.load(args.model_path))
    policy_criterion = CrossEntropyLoss()
    value_criterion = MSELoss()
    optimizer = Adam(net.parameters(), lr=args.learning_rate)

    # Train
    print('\n<Train Model>')
    for epoch in range(args.epochs):
        # adjust learning rate
        if epoch >= 50:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0005
        if epoch >= 80:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.00025
        
        net.train()
        train_loss = 0.0
        best_loss = 100.0
        for state, y_policy, y_value in tqdm(data_loader):
            state = state.to(device)
            y_policy = y_policy.to(device)
            y_value = y_value.to(device)
            
            optimizer.zero_grad()

            policy_output, value_output = net(state)
            value_output = value_output.squeeze()

            policy_loss = policy_criterion(policy_output, y_policy)
            value_loss = value_criterion(value_output, y_value)
            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(data_loader)
        if best_loss > train_loss:
            os.makedirs('./model/', exist_ok=True)
            torch.save(net.state_dict(), './model/latest.pth')

        print(f'{epoch+1}/{args.epochs}')
        print(f'    Loss: {train_loss:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reinforcement Train")

    # model hyper parameter
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--pv_eval_count', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--num_residual_block', type=int, default=16)
    parser.add_argument('--num_filters', type=int, default=128)

    # train hyper parameter
    parser.add_argument('--self_count', type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    train(args)