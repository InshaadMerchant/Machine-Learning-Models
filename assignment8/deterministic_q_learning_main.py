import time

from deterministic_q_learning import AgentModel_Q_Learning_Deterministic

# When you test your code, you can select the function arguments you 
# want to use by modifying the next lines

#environment_file = "environment1.txt"
environment_file = "environment2.txt"
actions_file = "actions2a.txt"
ntr = -0.04 # non_terminal_reward
gamma = 0.9
number_of_moves = 100000

start_time = time.time()
AgentModel_Q_Learning_Deterministic(environment_file, actions_file, 
                                    ntr, gamma, number_of_moves)

end_time = time.time()
print("\nPassive Q-Learning took %.2f seconds to run." % (end_time-start_time))
