import random
import numpy as np
import math
import copy

ROWS = 6
COLUMNS = 7
WINNER = False

class ConnectFour:
    def __init__(self, matrix = None):
        if matrix is not None:
            self.matrix = matrix.copy()
        else: 
            self.matrix = np.zeros((ROWS, COLUMNS), dtype = np.int8)
            #self.matrix = [[0 for _ in range(COLUMNS)] for _ in range(ROWS)]          
    
    def print_board_state(self):
        symbols = {1: 'A', -1: 'B', 0: '0'}
        for r in range(ROWS):
            print(' '.join(symbols[val] for val in self.matrix[r]))
            
    def get_valid_moves(self):
        #gives a list of columns that aren't full.
        return np.where(self.matrix[0] == 0)[0]
    
    def check_illegal_move(self,c):
        return self.matrix[0][c] != 0
    
    def drop_piece(self, player, c):
        if self.check_illegal_move(c):
                print(f"Column {c} is full!")
                return self.matrix
    
        #for r in range(ROWS - 1, -1, -1):
        #    if self.matrix[r][c] == 0:
        #        self.matrix[r][c] = piece
        #        break  # Stop searching once the piece is placed
        row = np.max(np.where(self.matrix[:, c] == 0)[0])
        self.matrix[row, c] = player        
        
        return self.matrix
    
    def check_winner(self, player):
        m = self.matrix
        #check rows
        for x in range(ROWS):
            for col in range(COLUMNS-3):
                if self.matrix[x][col] != 0 and np.all(m[x, col:col+4] == player):
                    #print (f'winner in row {x+1}')
                    return True
        #check columns
        for x in range(COLUMNS):
            for row in range(ROWS-3):
                if self.matrix[row][x] != 0 and np.all(m[row:row+4, x] == player):
                    #print (f'winner in column {x}')
                    return True
        #check diagonals top left to bottom right
        for x in range(COLUMNS-3):
            for row in range(ROWS-3):
                #if self.matrix[row][x] != 0 and self.matrix[row][x] == self.matrix[row+1][x+1] == self.matrix[row+2][x+2] == self.matrix[row+3][x+3]:
                if m[row][x] != 0 and np.all([m[row+i, x+i] == player for i in range(4)]):
                    #print (f'winner diagonal start at row {row+1} column{x+1}')
                    return True
        #check diagonals bottom left to top right
        for x in range(COLUMNS-3):
            for row in range(ROWS-3):
                #if m[row][x] != 0 and  np.all([m[row+3-i, x+i] == player for i in range(4)]):
                if np.all([m[row+3-i, x+i] == player for i in range(4)]):
                    #print (f'winner diagonal start at row {row+4} column{x+1}')
                    return True
        
        
        return False
    
    def check_draw(self, player):
        #True if the board is completely full (no valid moves left).
        return np.all(self.matrix[0] != 0)
    
class NODE:
    def __init__(self, move, parent, state, player_who_made_drop):
        self.move = move #move that lead to this state)
        self.parent = parent
        self.state = state
        self.N = 0
        self.Q = 0
        self.children = {}
        self.player_who_moved = player_who_made_drop
        #self.winner = False #true is terminal node
    
    def add_children(self, children: dict) -> None:
        for child in children:
            self.children[child.move] = child
        
    def value(self, explore: float = 1.41414): #1.414 = sqr(2)
        if self.N == 0:
            return 0 if explore == 0 else 9999999
        else:
            exploit = self.Q / self.N
            explore = explore * math.sqrt(math.log(self.parent.N) / self.N)
            return exploit + explore
        
class MCTS:
    def __init__(self, state, current_player):
        self.root_state = copy.deepcopy(state)
        self.root = NODE(None, None, copy.deepcopy(state), -current_player)
        self.run_time = 0 #for timer
        self.node_count = 0
        self.node_rollouts = 0
        self.player = current_player
        self.root_player = current_player
        
    def select_node(self):
        node = self.root
        state = copy.deepcopy(self.root_state)
        current_player = self.root_player
        #option 1 (perfect for Connect 4): Expand All Children First
        while len(node.children) != 0:
            children = node.children.values()
            max_value = max(child.value() for child in children)
            max_nodes = [x for x in children if x.value() == max_value]          
            node = random.choice(max_nodes)
            state.drop_piece(current_player, node.move)
            current_player = -current_player
            
            if node.N == 0:
                return node, state, current_player
            
        if self.expand(node, state, player):
            node =  random.choice(list(node.children.values()))
            state.drop_piece(player, node.move)
            current_player = -current_player
            
        return node, state, current_player
        
            
        #option2: Lazy Expansion (Standard MCTS Approach)
        #If a node has 7 legal moves in Connect 4, but only 3 are currently in node.children, select_node stops at this node so expand_node can pick one of the 4 remaining untried moves.
        #while self.is_fully_expanded(node) and not self.is_terminal(node):
            #node = max(node.children.values(), key=lambda n: n.value())
        #return node
    
    def expand(self, parent: NODE, state, current_player):
        last_player = -current_player
        if state.check_winner(last_player) or state.check_draw(last_player):
            return False
        valid_moves = state.get_valid_moves()
        children = [NODE(move, parent,copy.deepcopy(state), current_player) for move in valid_moves]
        parent.add_children(children)
        
        return True
    #def simulate(self, node: Node):
        
    def search(self, iterations=50):
        """Runs MCTS selection/expansion iterations and picks the best move."""
        for _ in range(iterations):
            # 1. Selection & 2. Expansion
            leaf_node, leaf_state, current_player = self.select_node()
    
            # 3. Simulation (Rollout)
            winner = self.simulation(leaf_state, player)
    
            # 4. Backpropagation
            self.backpropagate(leaf_node, winner)
    
        # Pick the child move with the highest visit count N
        best_child = max(self.root.children.values(), key=lambda c: c.N)
        return best_child.move
    
    def simulation(self, state, current_player):
        simulation_state = copy.deepcopy(state)
        simulation_player = current_player
        
        while True:
        # Check if the previous move caused a win
        # (sim_player * -1 was the player who just played)
            last_player = -simulation_player
            if simulation_state.check_winner(last_player):
                return last_player  # Return winning player (1 or -1)
    
            valid_moves = simulation_state.get_valid_moves()
            if len(valid_moves) == 0:
                return 0  # Draw
    
            # Make a random valid move
            move = random.choice(valid_moves)
            simulation_state.drop_piece(simulation_player, move)
    
            # Switch turns for the next simulation step
            simulation_player = -simulation_player
    def backpropagate(self, node, winner):
        #Propagates simulation outcome back up the tree path
        curr = node
        while curr is not None:
            curr.N += 1
            
            if winner == 0:
                curr.Q += 0.5  # Draw gives partial credit
            elif winner == curr.player_who_moved:
                curr.Q += 1.0  # Full reward if root player won
            else:
                curr.Q += 0.0  # No reward for a loss
    
            curr = curr.parent
        

#x = print(print_board_state(initialize_game()))
#print (random.randint(0,6))
game = ConnectFour()
game.print_board_state()
print ('-------')
#print (game.get_valid_moves())

while not WINNER:
    player = 1
    row_input = input("player 1, make a choice: ")
    game.drop_piece(1, int(row_input))
    game.print_board_state()
    WINNER = game.check_winner(1)
    DRAW = game.check_draw(1)
    if WINNER or DRAW:
        break
    print ('-------')
    player = -1
    print ("Player 2 (AI) is thinking ... ")
    mcts = MCTS(state=game, current_player = -1)
    ai_move = mcts.search(iterations = 1000)
    #valid_drop = random.choice(game.get_valid_moves())
    #game.drop_piece(-1, int(valid_drop))#random.randint(0,6))
    game.drop_piece(-1, ai_move)
    game.print_board_state()
    WINNER = game.check_winner(-1)
