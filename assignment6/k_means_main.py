from k_means import *


data_file = "toy_data/set1a.txt"
#data_file = "toy_data/set2a.txt"
#data_file = "toy_data/set2_1.txt"

K = 3
#initialization = "random"
initialization = "round_robin"


k_means(data_file, K, initialization)