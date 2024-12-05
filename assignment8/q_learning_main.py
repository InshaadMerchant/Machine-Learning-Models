import time

from q_learning import AgentModel_Q_Learning

# When you test your code, you can select the function arguments you 
# want to use by modifying the next lines

#environment_file = "environment1.txt"
environment_file = "environment2.txt"
moves_file = None
#ntr = -0.01 # non_terminal_reward
#gamma = 0.99
ntr = -0.04 # non_terminal_reward
gamma = 0.9

number_of_moves = 3000000
Ne = 50000

start_time = time.time()

AgentModel_Q_Learning(environment_file, ntr, gamma, number_of_moves, Ne)

end_time = time.time()

print("\nActive Q-Learning took %.2f seconds to run." % (end_time-start_time))