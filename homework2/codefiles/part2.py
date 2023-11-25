import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy import exp


def read_data(filename):
    with open(filename, 'r') as file:
            points = np.array([[float(x) for x in line.strip().split()] for line in file])
    return points

def rbf_kernel(p1, p2, gamma=0.1):
    return exp(-gamma * norm(p1 - p2) ** 2)

def k_means(points, k=2, max_iters=100):
    # centroids = np.random.choice(points, k, replace=False)
    centroids = [points[0], points[1]]

    for i in range(max_iters):
        clusters = [[] for _ in range(k)]
        for point in points:
            # assigning each point to the closest centroid
            distances = [norm(point - centroid) for centroid in centroids]
            closest_centroid = np.argmin(distances)
            clusters[closest_centroid].append(point)

        # updating the centroids
        new_centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters])
        if np.all(centroids == new_centroids): # no update => convergence
            break
        centroids = new_centroids

    return clusters, centroids



if __name__ == '__main__':
    points = read_data('kmeans_data.txt')
    # print(points.shape)
    
    plt.figure()
    for point in points:
        plt.scatter(point[0], point[1], color='blue')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Scatter plot of the dataset')
    plt.savefig('part2_input.png')
    
    # it can be observed that the data can be well divided into 2 clusters based on distance from the origin
    dist = np.array([np.sqrt(point[0] ** 2 + point[1] ** 2) for point in points])
    # print(dist.shape)
    
    _, centroids = k_means(dist, k=2)
    colors = ['red', 'green']
    # print(clusters.shape)
    cluster_ids = np.zeros(len(points), dtype=int)
    for i,point in enumerate(points):
        distances = [np.linalg.norm(np.sqrt(point[0] ** 2 + point[1] ** 2) - centroid) for centroid in centroids]
        cluster_ids[i] = np.argmin(distances)
    
    plt.figure()
    for i in range(len(points)):
        plt.scatter(points[i,0], points[i,1], color=colors[cluster_ids[i]])
    plt.title("K-means with distance as the parameter")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig('part2_1.png')
    
    L = 1 # Choosing only 1 landmark
    for i in range(10):
        print("iteration number ", i)
        landmark = points[np.random.choice(len(points))]
        landmark_based_data = np.array([rbf_kernel(point, landmark) for point in points])
        # print(landmark_based_data)
        clusters, centroids = k_means(landmark_based_data, k=2)
        # cluster_ids = np.zeros(len(points), dtype=int)
        # print(clusters.shape)
        # print(clusters)
        plt.figure()
        print(len(centroids))
        for j, cluster in enumerate(clusters):
            # print(cluster)
            cluster_points = points[[np.where(element == landmark_based_data)[0][0] for element in cluster]]
            print(cluster_points.shape)
            # print(cluster_points.shape)
            plt.scatter(cluster_points[:,0], cluster_points[:,1], color=colors[j])
        plt.scatter(landmark[0], landmark[1], color='blue')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Plot based on choosing a landmark")
        plt.savefig(f'part2_2_{i+1}_time.png')