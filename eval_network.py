import os
import torch

from pathlib import Path
from shutil import copy

from game import State, random_action, alpha_beta_action, mcts_action
from mcts import pv_mcts_action

def first_player_point(ended_state):
    # 1: win, 0: lose, 0.5: draw
    if ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    return 0.5

def play(next_actions):
    state = State()

    while True:
        if state.is_done():
            break

        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        action = next_action(state)

        state = state.next(action)

    return first_player_point(state)

def evaluate_algorithm(next_actions, eval_epochs):
    total_point = 0
    for eval_epoch in range(eval_epochs):
        if eval_epoch % 2 == 0:
            total_point += play(next_actions)
        else:
            total_point += 1 - play(list(reversed(next_actions)))

        print(f'\rEvaluate {eval_epoch+1}/{eval_epochs}')

    average_point = total_point / eval_epochs
    return average_point

def update_best_player():
    copy('./model/latest.pth', './model/best.pth')
    print('Change BestPlayer')

def evaluate_network(args, net, device):
    # Initialize the state
    state = State()

    # select action in mcts algorithm with latest & best model
    net.load_state_dict(torch.load('./model/latest.pth'))
    next_action_latest = pv_mcts_action(net, state, device, args.pv_eval_count, args.temperature)
    net.load_state_dict(torch.load('./model/best.pth'))
    next_action_best = pv_mcts_action(net, state, device, args.pv_eval_count, args.temperature)
    next_actions = (next_action_latest, next_action_best)

    # calculate the average point and change the model
    average_point = evaluate_algorithm(next_actions, args.eval_epochs)
    if average_point > 0.5:
        update_best_player()
        return True
    else:
        return False

def evaluate_best_player(args, net, device):
    net.load_state_dict(torch.load('./model/latest.pth'))
    state = State()

    # mcts action function
    next_pv_mcts_action = pv_mcts_action(net, state, device, 0.0)

    # vs. random algorithm
    next_actions = (next_pv_mcts_action, random_action)
    average_point = evaluate_algorithm(next_actions, args.eval_epochs)
    print(f'VS_Random: {average_point}')

    # vs. alpha_beta algorithm
    next_actions = (next_pv_mcts_action, alpha_beta_action)
    average_point = evaluate_algorithm(next_actions, args.eval_epochs)
    print(f'VS_AlphaBeta: {average_point}')

    # vs. native mcts algorithm
    next_actions = (next_pv_mcts_action, mcts_action)
    average_point = evaluate_algorithm(next_actions, args.eval_epochs)
    print(f'VS_MCTS: {average_point}')