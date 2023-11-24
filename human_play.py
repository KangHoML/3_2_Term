import tkinter as tk
import torch
import argparse

from game import State, random_action
from mcts import pv_mcts_action
from dual_network import DualNetwork
from pathlib import Path
from threading import Thread
from PIL import Image, ImageTk

parser = argparse.ArgumentParser()
parser.add_argument('--wxh', type=int, default=240)
parser.add_argument('--num_residual_block', type=int, default=16)
parser.add_argument('--num_filters', type=int, default=128)
parser.add_argument('--pv_eval_count', type=int, default=50)
parser.add_argument('--temperature', type=float, default=1.0)
args = parser.parse_args()

net = DualNetwork(num_residual_block=args.num_residual_block, num_filters=args.num_filters)
net.load_state_dict(torch.load('./model/best.pth', map_location = torch.device('cpu')))
net.eval()

class GameUI(tk.Frame):
    def __init__(self, master=None, net=None, args=args):

        tk.Frame.__init__(self,master)
        self.master.title('Tic Tac Toe')
        
        self.state = State()
        
        self.next_action = pv_mcts_action(net, self.state, device = torch.device('cpu'), pv_eval_count=args.pv_eval_count, temperature = 0.0)
        
        self.c = tk.Canvas(self, width = args.wxh, height = args.wxh, highlightthickness = 0)
        # have to change
        self.c.bind('<Button-1>', self.turn_of_human)
        self.c.pack()
        
        self.on_draw()
        
        # human turn
    def turn_of_human(self, event):
        # game done
        if self.state.is_done():
            self.state = State()
            self.on_draw()
            return

        if not self.state.is_first_player():
            return
        
        area = args.wxh / 3
        x = int(event.x / area)      # 240 / 80 => 3 => 3x3
        y = int(event.y / area)
        if x < 0 or 2 < x or y < 0 or 2 < y:  
            return
        action = x + y * 3
        
        if not (action in self.state.legal_actions()):
            return

        self.state = self.state.next(action)
        self.on_draw()

       
        self.master.after(1, self.turn_of_ai)

    # AI turn
    def turn_of_ai(self):
        # game done
        if self.state.is_done():
            return
        
        action = self.next_action(self.state)

        self.state = self.state.next(action)
        self.on_draw()
    
    # Draw Pieces
    def draw_piece(self, index, first_player):
        x = (index % 3) * 80 + 10
        y = int(index / 3) * 80 + 10
        if first_player:
            self.c.create_oval(x, y, x + 60, y + 60, width=2.0, outline='#FFFFFF') #o
        else:
            self.c.create_line(x, y, x + 60, y + 60, width=2.0, fill='#5D5D5D') #x
            self.c.create_line(x + 60, y, x, y + 60, width=2.0, fill='#5D5D5D')

    # On_draw
    def on_draw(self):
        self.c.delete('all')
        self.c.create_rectangle(0, 0, args.wxh, args.wxh, width=0.0, fill='#01DF01')
        self.c.create_line(args.wxh/3, 0, args.wxh/3, args.wxh, width=2.0, fill='#000000')
        self.c.create_line(args.wxh/3*2, 0, args.wxh/3*2, args.wxh, width=2.0, fill='#000000')
        self.c.create_line(0, args.wxh/3, args.wxh, args.wxh/3, width=2.0, fill='#000000')
        self.c.create_line(0, args.wxh/3*2, args.wxh, args.wxh/3*2, width=2.0, fill='#000000')
        for i in range(9):
            if self.state.pieces[i] == 1:
                self.draw_piece(i, self.state.is_first_player())
            if self.state.enemy_pieces[i] == 1:
                self.draw_piece(i, not self.state.is_first_player())


f = GameUI(net=net)
f.pack()
f.mainloop()