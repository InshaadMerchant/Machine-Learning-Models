import numpy as np

def load_environment(filename):
    environment = []
    with open(filename, 'r') as file:
        for line in file:
            environment.append(line.strip().split(","))
    return environment

def load_actions(filename):
    actions = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().replace(',', '').split()
            actions.append((int(parts[0]), int(parts[1]), parts[2], int(parts[3]), int(parts[4])))
    return actions

def get_reward(cell, non_terminal_reward):
    if cell == 'X':
        return 0.0  # Assuming 'X' is blocked
    elif cell == '1.0':
        return 1.0  # Terminal state positive
    elif cell == '-1.0':
        return -1.0  # Terminal state negative
    elif cell == 'I' or cell == '0':
        return non_terminal_reward  # Initial or non-terminal state
    try:
        return float(cell)  # Convert numeric strings to float
    except ValueError:
        return non_terminal_reward  # Default non-terminal reward for other non-numeric states

def AgentModel_Q_Learning_Deterministic(environment_file, actions_file, non_terminal_reward, gamma, number_of_moves):
    environment = load_environment(environment_file)
    actions = load_actions(actions_file)
    utilities = np.zeros_like(environment, dtype=float)

    # Initialize utilities
    for i in range(len(environment)):
        for j in range(len(environment[i])):
            utilities[i][j] = get_reward(environment[i][j], non_terminal_reward)

    # Process actions using specified results
    for count, (row, col, action, next_row, next_col) in enumerate(actions):
        if count >= number_of_moves:
            break
        if environment[row][col] not in ["X", "1.0", "-1.0", "I"]:  # Skip processing for special states
            current_utility = utilities[row][col]
            reward = get_reward(environment[next_row][next_col], non_terminal_reward)
            utilities[next_row][next_col] = reward + gamma * current_utility  # Bellman update

    print("Q-Learning Deterministic: ntr = {:.4f}, gamma = {:.4f}, moves = {}".format(non_terminal_reward, gamma, number_of_moves))
    print("utilities:")
    for row in utilities:
        print("  ".join(f"{value:.3f}" for value in row))
    print()

# Uncomment the following line to test or execute in your script
# AgentModel_Q_Learning_Deterministic('environment2.txt', 'actions2a.txt', -0.04, 0.9, 100000)
