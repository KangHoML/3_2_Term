import os
import torch
import pickle
import numpy as np

from datetime import datetime
from pathlib import Path
from game import State
from mcts import pv_mcts_scores
from dual_network import DualNetwork

# value of first player (1 : win / 0 : draw / -1 : lose)
def first_player_value(ended_state):
    if ended_state.is_lose():
        return -1 if ended_state.is_first_player() else 1
    return 0

# store the train data
def write_data(history):
    now = data



def play(net):
    history = []