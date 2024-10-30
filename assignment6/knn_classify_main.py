from knn_classify import *


# When you test your code, you can change this line to reflect where the 
# dataset directory is located on your machine.
dataset_directory = "./uci_data"

# When you test your code, you can select the dataset you want to use 
# by modifying the next lines
dataset = "pendigits"
#dataset = "satellite"
#dataset = "yeast"


training_file = dataset_directory + "/" + dataset + "_training.txt"
test_file = dataset_directory + "/" + dataset + "_test.txt"

# When you test your code, you can select the function arguments you want to use 
# by modifying the next lines
k = 3

knn_classify(training_file, test_file, k)