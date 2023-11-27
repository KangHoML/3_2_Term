import torch
import argparse
import tkinter as tk

from game import State
from dual_network import DualNetwork
from mcts import pv_mcts_scores, pv_mcts_action
from self_play import first_player_value, write_data

class GameUI(tk.Frame):
    def __init__(self, net, size, pv_eval_count, temperture, game_count = None, master=None):
        tk.Frame.__init__(self, master)
        
        # about UI
        self.master.title('Tic Tac Toe')
        self.size = size
        self.game_count = game_count
        self.current_count = 0

        self.net = net
        self.net.load_state_dict(torch.load('./model/best.pth'))
        net.eval()

        self.pv_eval_count = pv_eval_count
        self.temperature  = temperture
        self.next_action = pv_mcts_action(self.net, pv_eval_count=self.pv_eval_count, temperature = 0.0)
        
        self.c = tk.Canvas(self, width=self.size, height=self.size, highlightthickness=0)
        self.c.bind('<Button-1>', lambda event : self.turn_of_human(event))
        self.c.pack()

        self.history = []
        self.current_history = []
        self.state = State()
        self.on_draw()

    def turn_of_human(self, event):
        if self.state.is_done():
            self.reset_game()
            return
        
        x = int(event.x / (self.size/3))
        y = int(event.y / (self.size/3))
        
        if x < 0 or 2 < x or y < 0 or 2 < y:  
            return
        
        action = x + y * 3
        if not (action in self.state.legal_actions()):
            return
        
        self.state = self.state.next(action)
        self.on_draw()
        self.master.after(1, self.turn_of_ai)

    def turn_of_ai(self):
        if self.state.is_done():
            self.reset_game()
            return
        
        scores = pv_mcts_scores(self.net, self.state, self.pv_eval_count, self.temperature)

        # current history
        policies = [0] * 9
        for action, policy in zip(self.state.legal_actions(), scores):
            policies[action] = policy
        self.current_history.append([[self.state.pieces, self.state.enemy_pieces], policies, None])

        action = self.next_action(self.state)
        self.state = self.state.next(action)
        self.on_draw()

    def reset_game(self):
        # update history
        value = first_player_value(self.state)
        for i in range(len(self.current_history)):
            self.current_history[i][2] = value
            value = -value
        self.history.extend(self.current_history)
        print(self.history)

        # initialize state & current history
        self.current_history = []
        self.state = State()
        self.on_draw()

        # check game_count & write history
        if self.game_count is not None:
            self.current_count += 1

            if self.current_count >= self.game_count:
                write_data(self.history)
                self.master.destroy()
    
    def draw_piece(self, idx, is_first_player):
        x = (idx % 3) * 80 + 10
        y = int(idx / 3) * 80 + 10

        if is_first_player:
            self.c.create_oval(x, y, x + 60, y + 60, width=2.0, outline='#FFFFFF') # O
        else:
            # X
            self.c.create_line(x, y, x + 60, y + 60, width=2.0, fill='#5D5D5D')
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
                self.draw_piece(i, self.state.is_first_player())
            if self.state.enemy_pieces[i] == 1:
                self.draw_piece(i, not self.state.is_first_player())

parser = argparse.ArgumentParser('Game UI')
parser.add_argument('--game_count', type=int, default=None)
parser.add_argument('--size', type=int, default=240)
parser.add_argument('--num_residual_block', type=int, default=16)
parser.add_argument('--num_filters', type=int, default=128)
parser.add_argument('--pv_eval_count', type=int, default=50)
parser.add_argument('--temperature', type=float, default=1.0)

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = DualNetwork(num_residual_block=args.num_residual_block, num_filters=args.num_filters).to(device)
    
    if args.game_count is not None:
        f = GameUI(net, args.size, args.pv_eval_count, args.temperature, game_count=args.game_count)
    else:
        f = GameUI(net, args.size, args.pv_eval_count, args.temperature)
    f.pack()
    f.mainloop()
