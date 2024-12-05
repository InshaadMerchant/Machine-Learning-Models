import numpy as np
from random import random

class Action:
    arrows = ['^','<','v','>']
    names = ['up','left','down','right']
    translate = [(1,0),(0,-1),(-1,0),(0,1)]

    def __init__(self, direction: str):
        direction = direction.lower()
        if direction not in Action.names:
            raise ValueError('Invalid action')
        
        self.direction = Action.names.index(direction)

    def ApplyTo(self, state):
        translation = Action.translate[self.direction]
        Ty = translation[0]
        Tx = translation[1]
        return (state[0]+Ty, state[1]+Tx)
    
    def RotateLeft(self):
        return Action(Action.names[(self.direction+1)%4])
    
    def RotateRight(self):
        return Action(Action.names[(self.direction-1)%4])

    def Actions():
        return [Action(name) for name in Action.names]

    def __eq__(self, other):
        if other == None:
            return False
        return self.direction == other.direction
    
    def __hash__(self):
        return hash(self.direction)
    
    def __str__(self):
        return Action.arrows[self.direction]
    
    def __repr__(self):
        return Action.arrows[self.direction]

class Environment:
    def __init__(self, data_file, ntr):
        obstacles = []
        terminals = {}

        with open(data_file, 'r') as filestream:
            rows = filestream.readlines()
            num_rows = len(rows)
            num_cols = len(rows[0].split(','))

            for y, row in enumerate(rows):
                for x, val in enumerate(row.split(',')):
                    pos = (num_rows-y, x+1)
                    val = val.strip()

                    if val == 'X':      # obstacle
                        obstacles.append(pos)
                    elif val == 'I':    # start
                        start_state = pos
                    elif val != '.':    # terminal
                        terminals[pos] = float(val)
        
        self.obstacles = obstacles
        self.terminals = terminals
        self.start_state = start_state
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.ntr = ntr
    
    def reward(self, state):
        if state in self.obstacles:
            raise ValueError('Can not be inside an obstacle')
        elif state in self.terminals:
            return self.terminals[state]
        else:
            return self.ntr

    def ExecuteAction(self, state, action: Action):
        rand = random()
        if rand < .1:
            action = action.RotateLeft()
        elif rand > .9:
            action = action.RotateRight()
        
        (y, x) = action.ApplyTo(state)
        if (y, x) in self.obstacles: # if hitting obstacle
            return state
        elif (y <= 0 or y > self.num_rows) or (x <= 0 or x > self.num_cols): # if hitting side of map
            return state
        return (y, x)

def AgentModel_Q_Learning(environment_file, ntr, gamma, number_of_moves, Ne):
    # Initialize environment
    env = Environment(environment_file, ntr)
    
    # Q and N dictionaries to track Q-values and action counts
    Q = {}  # Q values for (s,a)
    N = {}  # # of occurrences of (s,a)
    total_actions_executed = 0

    # Main Q-Learning algorithm
    while total_actions_executed < number_of_moves:
        # Reset actor to start state
        cur_location = env.start_state
        s, r, a = None, None, None  # previous state, reward, action taken

        while total_actions_executed < number_of_moves:
            # Sense current state and reward
            s_p, r_p = cur_location, env.reward(cur_location)

            # Q-Learning Update
            if s_p in env.terminals:
                Q[(s_p, None)] = r_p
            
            if s is not None:
                N[(s,a)] = N.get((s,a), 0) + 1
                c = 20 / (19 + N[(s,a)])

                prev_Q = (1-c) * Q.get((s,a), 0)
                new_Q = c * (r + gamma * max([Q.get((s_p,action), -1) for action in Action.Actions()+[None]]))
                Q[(s,a)] = prev_Q + new_Q

            # Terminal state check
            if s_p in env.terminals:
                break

            # Action selection with exploration
            f_vals = {}
            for action in Action.Actions():
                f_vals[action] = (Q.get((s_p,action), 0) if N.get((s_p,action), 0) >= Ne else 1)
            
            max_v = max(f_vals.values())
            candidate_actions = [key for key, val in f_vals.items() if val == max_v]
            a = candidate_actions[np.random.randint(len(candidate_actions))]

            # Execute action
            new_location = env.ExecuteAction(cur_location, a)
            cur_location = new_location
            total_actions_executed += 1

            s, r = s_p, r_p

    # Output results
    utilities = np.zeros((env.num_rows, env.num_cols))
    policy = np.empty((env.num_rows, env.num_cols), dtype=str)

    for i in range(env.num_rows):
        for j in range(env.num_cols):
            state = (i+1,j+1)

            # Compute utility for the state
            utility = max([Q.get((state,action), -1) for action in Action.Actions()+[None]])
            
            # Determine best action for the state
            if state in env.terminals:
                policy[i,j] = 'o'
                utilities[i,j] = env.terminals[state]
            elif state in env.obstacles:
                policy[i,j] = 'x'
                utilities[i,j] = 0
            else:
                best_action_index = np.argmax([Q.get((state,action), 0) for action in Action.Actions()])
                policy[i,j] = Action(Action.names[best_action_index])
                utilities[i,j] = utility

    # Print utilities
    print('utilities:')
    for row in np.flip(utilities, axis=0):
        for val in row:
            print('%6.3f ' % val, end='')
        print()
    print()

    # Print policy
    print('policy:')
    for row in np.flip(policy, axis=0):
        for val in row:
            print('%6s ' % val, end='')
        print()