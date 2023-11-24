import torch
import numpy as np

from math import sqrt
from pathlib import Path
from game import State
from dual_network import DualNetwork

def predict(net, state, device):
    x = torch.tensor([state.pieces, state.enemy_pieces], dtype=torch.float32).to(device)
    x = x.view(2, 3, 3).permute(1, 2, 0).unsqueeze(0)
    
    net.eval()
    with torch.no_grad():
        policies, value = net(x)
        policies = policies.view(-1)
        value = value.item()

    policies = policies[state.legal_actions()]
    policies /= policies.sum() if policies.sum() else 1

    return policies, value
        
# transform node list to score list
def nodes_to_scores(nodes):
    scores = []
    for c in nodes:
        scores.append(c.n)
    return scores

class Node:
    def __init__(self, state, net, device, p):
        self.state = state
        self.net = net
        self.device = device
        self.p = p
        self.w = 0
        self.n = 0
        self.child_nodes = None

    # calcualte the value
    def evaluate(self):
        if self.state.is_done():
            value = -1 if self.state.is_lose() else 0
            self.w += value
            self.n += 1
            return value
        
        if not self.child_nodes:
            policies, value = predict(self.net, self.state, self.device)
            self.w += value
            self.n += 1

            self.child_nodes = []
            for action, policy in zip(self.state.legal_actions(), policies):
                self.child_nodes.append(Node(self.state.next(action), self.net, self.device, policy))
            return value
        
        else:
            value = -self.next_child_node().evaluate()
            self.w += value
            self.n += 1
            return value
    
    # select child node with arc score
    def next_child_node(self):
        C_PUCT = 1.0
        t = sum(nodes_to_scores(self.child_nodes))
        pucb_values = []
        for child_node in self.child_nodes:
            # pucb_values.append((-child_node.w / child_node.n + C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n) if child_node.n else C_PUCT * child_node.p * sqrt(t)))
            pucb_values.append((-child_node.w / child_node.n if child_node.n else 0.0) + C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n))

        pucb_values = torch.tensor(pucb_values, device=self.device)
        return self.child_nodes[torch.argmax(pucb_values).item()] # select max arc score in child node

def pv_mcts_scores(net, state, device, temperature, pv_eval_count):
    root_node = Node(state, net, device, 0)

    for _ in range(pv_eval_count):
        root_node.evaluate()

    scores = nodes_to_scores(root_node.child_nodes)
    if temperature == 0:
        action = torch.argmax(scores).item()
        scores = torch.zeros(len(scores), device=device)
        scores[action] = 1
    else:
        scores = boltzman(scores, temperature)
    
    return scores

def pv_mcts_action(net, state, device, pv_eval_count, temperature=0):
    scores = pv_mcts_scores(net, state, device, temperature, pv_eval_count)
    action = torch.multimnomial(scores, 1).item()
    return state.legal_actions()[action]

def boltzman(xs, temperature):
    xs = torch.tensor(xs, dtype=torch.float32)
    xs = torch.exp(xs / temperature)
    return xs / xs.sum()

if __name__ == '__main__':
    net = DualNetwork(num_residual_block=16, num_filters=128)
    net.eval()
    state = State()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    while True:
        if state.is_done():
            break
        action = pv_mcts_action(net, state, device, pv_eval_count=50, temperature=1.0)
        state = state.next(action)

        print(state)