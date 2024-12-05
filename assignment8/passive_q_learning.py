import numpy as np
import copy

def policy_evaluation(states, transition_probability, reward, gamma, policy, iterations, utilities):
    current_utilities = copy.deepcopy(utilities)  # Make a copy of the initial utilities
    for k in range(1, iterations + 1):
        previous_utilities = copy.deepcopy(current_utilities)  # Copy the utilities from the previous iteration
        for state in states:
            if policy[state[0]][state[1]] not in ['>', '^', '<', 'v']:  # Only process valid actions
                continue
            action = ['>', '^', '<', 'v'].index(policy[state[0]][state[1]])  # Find the action index
            # Sum over all possible next states
            current_utilities[state] = reward(state) + gamma * sum(transition_probability(next_state, state, action) * previous_utilities[next_state] for next_state in states)
    return current_utilities

def AgentModel_Q_Learning_Passive(environment_file, policy_file, non_terminal_reward, gamma, number_of_moves):
    environment_map = []
    policy_map = []

    with open(environment_file, 'r') as file:
        for line in file:
            environment_map.append(line.strip().split(","))

    with open(policy_file, 'r') as file:
        for line in file:
            policy_map.append(line.strip().split(","))

    utilities = np.zeros((len(environment_map), len(environment_map[0])), dtype=float)

    def transition_probability(next_state, state, action):
        directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        if not (0 <= next_state[0] < len(environment_map) and 0 <= next_state[1] < len(environment_map[0])):
            return 0.0
        intended = directions[action]
        # Success in intended direction
        if next_state == (state[0] + intended[0], state[1] + intended[1]):
            return 0.8
        # Failure, moves in perpendicular directions
        side_directions = [(state[0] + d[0], state[1] + d[1]) for d in [(-1, 0), (1, 0)] if d != intended]
        if next_state in side_directions:
            return 0.1
        return 0.0

    def reward(state):
        cell = environment_map[state[0]][state[1]]
        if cell == "1.0":
            return 1.0
        elif cell == "-1.0":
            return -1.0
        else:
            return non_terminal_reward  # Reward for non-terminal states

    states = [(i, j) for i in range(len(environment_map)) for j in range(len(environment_map[0])) if environment_map[i][j] not in ["X", "o"]]
    # Initialize terminal states
    for i in range(len(environment_map)):
        for j in range(len(environment_map[i])):
            if environment_map[i][j] == "1.0":
                utilities[i][j] = 1.0
            elif environment_map[i][j] == "-1.0":
                utilities[i][j] = -1.0

    utilities = policy_evaluation(states, transition_probability, reward, gamma, policy_map, number_of_moves, utilities)

    print(f"Q-Learning Passive: ntr = {non_terminal_reward:.4f}, gamma = {gamma:.4f}, moves = {number_of_moves}")
    print("utilities:")
    for row in utilities:
        print("  " + "  ".join(f"{x:.3f}" for x in row))
    print()

