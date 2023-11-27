import os
import torch
import pickle
import argparse
import tkinter as tk

from datetime import datetime
from threading import Thread
from pathlib import Path

from game import State
from dual_network import DualNetwork
from mcts import pv_mcts_scores

class GameUI(tk.Frame):
    def __init__(self, master, size, net):
        tk.Frame.__init__(self, master)

        self.master.title('Tic Tac Toe')
        self.size = size
        self.net = net.load_state_dict('./model/best.pth')

        # widget
        self.c = tk.Canvas(self, width=self.size, height=self.size, highlightbackground=0)
        self.c.bind('<Button-1>', lambda event : self.play(event, net))
        self.c.pack()
        
    def play(self, event, net):
        if self.state.is_done():
            self.state = State()


    def draw_piece(self, idx):
        x = (idx % 3) * 80 + 10
        y = int(idx / 3) * 80 + 10

        if self.state.is_first_player():
            self.c.create_oval(x, y, x + 60, y + 60, width=2.0, outline='#FFFFFF') # O
        else:
            self.c.create_line(x, y, x + 60, y + 60, width=2.0, fill='#5D5D5D') # X
            self.c.create_line(x + 60, y, x, y + 60, width=2.0, fill='#5D5D5D')

    def on_draw(self):
        # draw board
        self.c.delete('all')
        self.c.create_rectangle(0, 0, self.size, self.size, width=0.0, fill='#01DF01')
        self.c.create_line(self.size/3, 0, self.size/3, self.size, width=2.0, fill='#000000')
        self.c.create_line(self.size/3*2, 0, self.size/3*2, self.size, width=2.0, fill='#000000')
        self.c.create_line(0, self.size/3, self.size, self.size/3, width=2.0, fill='#000000')
        self.c.create_line(0, self.size/3*2, self.size, self.size/3*2, width=2.0, fill='#000000')

        # draw piece
        for i in range(9):
            if self.state.pieces[i] == 1:
                self.draw_piece(i)
            if self.state.enemy_pieces[i] == 1:
                self.draw_piece(i)

    def destroy_ui(self)
