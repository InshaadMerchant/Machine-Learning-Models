#Inshaad Merchant --- 1001861293
import numpy as np

def value_iteration(environment_file, Reward, gamma, k):
    
    with open(environment_file, 'r') as f: # Read the environment file
        grid = []
        for line in f:
            
            row = line.strip().split(',')
            row = [cell.strip() for cell in row if cell.strip()]
            
            if row:
                grid.append(row)
    
    # Get grid dimensions
    height = len(grid)
    width = len(grid[0])
    
    # Initial utility values will be all zero
    U = np.zeros((height, width))  # Current utilities
    R = np.zeros((height, width))  # Rewards matrix
    
    # Initialize blocked and terminal states
    terminal_states = set()
    blocked = set()
    
    # Parse the environment file
    for i in range(height):
        for j in range(width):
            cell = grid[i][j]
            if cell == '.':
                R[i][j] = Reward
            elif cell == 'X':
                blocked.add((i, j))
                R[i][j] = 0
            else:  # Terminal state
                terminal_states.add((i, j))
                R[i][j] = float(cell)
                U[i][j] = float(cell) 
    
    # List down the Possible actions
    actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    action_symbols = {
        (-1, 0): '^',  # up
        (0, 1): '>',   # right
        (1, 0): 'v',   # down
        (0, -1): '<'   # left
    }
    
    # Main value iteration loop
    for i in range(k):
        next_utilities = np.copy(U)
        
        for i in range(height):
            for j in range(width):
                if (i, j) in terminal_states or (i, j) in blocked:
                    continue
                
                # Calculate maximum expected utility over all actions
                max_utility = float('-inf')
                
                for action in actions:
                    utility = 0
                    
                    # Main intended direction (probability 0.8)
                    ni, nj = i + action[0], j + action[1]
                    if 0 <= ni < height and 0 <= nj < width and (ni, nj) not in blocked:
                        utility += 0.8 * U[ni, nj]
                    else:
                        utility += 0.8 * U[i, j]
                    
                    # Left of intended direction (probability 0.1)
                    left_action = (action[1], -action[0])  # Rotate 90° left
                    ni, nj = i + left_action[0], j + left_action[1]
                    if 0 <= ni < height and 0 <= nj < width and (ni, nj) not in blocked:
                        utility += 0.1 * U[ni, nj]
                    else:
                        utility += 0.1 * U[i, j]
                    
                    # Right of intended direction (probability 0.1)
                    right_action = (-action[1], action[0])  # Rotate 90° right
                    ni, nj = i + right_action[0], j + right_action[1]
                    if 0 <= ni < height and 0 <= nj < width and (ni, nj) not in blocked:
                        utility += 0.1 * U[ni, nj]
                    else:
                        utility += 0.1 * U[i, j]
                    
                    max_utility = max(max_utility, utility)
                
                next_utilities[i, j] = R[i, j] + gamma * max_utility # Bellman update
        
        U = next_utilities
    
    # Calculate optimal policy
    policy = [['' for _ in range(width)] for _ in range(height)]
    
    for i in range(height):
        for j in range(width):
            if (i, j) in terminal_states:
                policy[i][j] = 'o'
            elif (i, j) in blocked:
                policy[i][j] = 'x'
            else:
                # Find best action
                best_action = None
                max_utility = float('-inf')
                
                for action in actions:
                    utility = 0
                    
                    # Calculate expected utility for each action
                    # Main direction (0.8 probability)
                    ni, nj = i + action[0], j + action[1]
                    if 0 <= ni < height and 0 <= nj < width and (ni, nj) not in blocked:
                        utility += 0.8 * U[ni, nj]
                    else:
                        utility += 0.8 * U[i, j]
                    
                    # Left of intended direction (0.1 probability)
                    left_action = (action[1], -action[0])
                    ni, nj = i + left_action[0], j + left_action[1]
                    if 0 <= ni < height and 0 <= nj < width and (ni, nj) not in blocked:
                        utility += 0.1 * U[ni, nj]
                    else:
                        utility += 0.1 * U[i, j]
                    
                    # Right of intended direction (0.1 probability)
                    right_action = (-action[1], action[0])
                    ni, nj = i + right_action[0], j + right_action[1]
                    if 0 <= ni < height and 0 <= nj < width and (ni, nj) not in blocked:
                        utility += 0.1 * U[ni, nj]
                    else:
                        utility += 0.1 * U[i, j]
                    
                    if utility > max_utility:
                        max_utility = utility
                        best_action = action
                
                policy[i][j] = action_symbols[best_action]
    
    # Print utilities
    print("utilities:")
    for i in range(height):
        print(" ".join(["{:6.3f}".format(U[i][j]) for j in range(width)]))
    
    # Print policy
    print("\npolicy:")
    for i in range(height):
        print(" ".join(policy[i]))

    return U