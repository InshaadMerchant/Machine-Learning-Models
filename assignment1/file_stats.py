import sys
import numpy as np

def file_stats(filename):
    data = np.loadtxt(filename)

    #Calculating Mean and Standard Deviation
    mean = np.mean(data, axis=0)
    st_dev = np.std(data, axis=0, ddof=1)  

    #Printing out the results
    for index, (mean, std) in enumerate(zip(mean, st_dev), 1):
        print(f"Column {index}: mean = {mean:.4f}, std = {std:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("filename not provided")
    else:
        filename = sys.argv[1]
        file_stats(filename)
        