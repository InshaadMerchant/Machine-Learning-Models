#Inshaad Merchant ----------- 1001861293
import numpy as np

def read_data(data_file: str) -> np.ndarray:
    data = []
    with open(data_file, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()] # Split line by whitespace and convert to float
            data.append(values)
    return np.array(data)

def initialize_clusters(data: np.ndarray, K: int, method: str) -> np.ndarray:
    N = len(data)
    
    if method == "random": # Random assignment of points to clusters (1 to K)
        return np.random.randint(1, K + 1, size=N)
    
    elif method == "round_robin":  # Round-robin assignment: 1,2,3,1,2,3,...
        return np.array([((i % K) + 1) for i in range(N)])
    
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
        min_dist = float('inf')
        best_cluster = None
        
        point = data[i].reshape(-1)  # Ensure point is 1D array
        
        for k in range(K):
            centroid = centroids[k].reshape(-1)  # Ensure centroid is 1D array
            dist = np.sum((point - centroid) ** 2)  # Euclidean distance squared
            
            if dist < min_dist:
                min_dist = dist
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

def k_means(data_file: str, K: int, initialization: str):
    data = read_data(data_file) # Read data
    
    cluster_assignments = initialize_clusters(data, K, initialization) # Initialize cluster assignments
    
    prev_assignments = None # Keep track of previous assignments to check for convergence
    
    while not np.array_equal(cluster_assignments, prev_assignments):
        
        prev_assignments = cluster_assignments.copy() # Store current assignments
        centroids = compute_centroids(data, cluster_assignments, K) # Compute centroids
        cluster_assignments = assign_clusters(data, centroids) # Assign points to nearest centroids
    
    print_results(data, cluster_assignments) # Print final cluster assignments