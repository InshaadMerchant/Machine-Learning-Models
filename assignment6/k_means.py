#Inshaad Merchant ----------- 1001861293
import numpy as np
import pandas as pd

def initialize_clusters(data: np.ndarray, K: int, method: str) -> np.ndarray:
    N = len(data)
    assignments = np.zeros(N, dtype=int)
    
    if method == "random":  # Random assignment of points to clusters (1 to K)
        return np.random.randint(1, K + 1, size=N)
    
    elif method == "round_robin":  # Round-robin assignment: 1,2,3,1,2,3,...
        for i in range(N):
            assignments[i] = (i % K) + 1  # This ensures proper 1-based cluster numbering
        return assignments
    
    else:
        raise ValueError(f"Unknown initialization method: {method}")

def compute_centroids(data: np.ndarray, cluster_assignments: np.ndarray, K: int) -> np.ndarray:
    dimension = data.shape[1] if len(data.shape) > 1 else 1
    centroids = np.zeros((K, dimension))
    
    for k in range(1, K + 1):
        # Get all points assigned to cluster k
        mask = (cluster_assignments == k)
        if np.any(mask):
            if dimension == 1:
                centroids[k-1] = np.mean(data[mask])
            else:
                centroids[k-1] = np.mean(data[mask], axis=0)
    
    return centroids

def assign_clusters(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    N = len(data)
    K = len(centroids)
    assignments = np.zeros(N, dtype=int)
    
    for i in range(N):
        min_distance = float('inf')
        best_cluster = None
        
        point = data[i].reshape(-1)  # Ensure point is 1D array
        
        for k in range(K):
            centroid = centroids[k].reshape(-1)  # Ensure centroid is 1D array
            distance = np.sum((point - centroid) ** 2)  # Euclidean distance squared
            
            if distance < min_distance:
                min_distance = distance
                best_cluster = k + 1  # Clusters are 1-indexed
        
        assignments[i] = best_cluster
    
    return assignments

def print_results(data: np.ndarray, cluster_assignments: np.ndarray):
    for i in range(len(data)):
        if len(data.shape) == 1 or data.shape[1] == 1:
            # 1D data
            print('%10.4f --> cluster %d' % (data[i], cluster_assignments[i]))
        else:
            # 2D data
            print('(%10.4f, %10.4f) --> cluster %d' % 
                  (data[i,0], data[i,1], cluster_assignments[i]))

def k_means(toy_data: str, K: int, initialization: str):
    # Read data 
    data = pd.read_csv(toy_data, sep='\s+', header=None).to_numpy()
    
    cluster_assignments = initialize_clusters(data, K, initialization)  # Initialize cluster assignments
    
    print("Initial assignments:")
    print_results(data, cluster_assignments)
    
    previous_assignments = None  # Keep track of previous assignments to check for convergence
    
    while not np.array_equal(cluster_assignments, previous_assignments):
        previous_assignments = cluster_assignments.copy()  # Store current assignments
        centroids = compute_centroids(data, cluster_assignments, K)  # Compute centroids
        cluster_assignments = assign_clusters(data, centroids)  # Assign points to nearest centroids
    
    print("\nFinal assignments after convergence:")
    print_results(data, cluster_assignments)  # Print final cluster assignments