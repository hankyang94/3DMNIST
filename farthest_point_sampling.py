import math
import random
import numpy as np


def img_to_point_cloud(img,k):
    # Given an MNIST image, obtain a 3D point cloud
    # Assume that the z coordinates of the point is 0
    img[img<0.5] = 0 # first remove blurred pixels
    non_zero_coord = list(np.transpose(np.nonzero(img)))
    if len(non_zero_coord) < k:
        print('No enough non-zero values in img.')
        return None
    else:
        # use farthest sampling to get downsampled coordinates
        cloud_2D = np.array(incremental_farthest_search(non_zero_coord,k))
        # center the point cloud
        mean = np.mean(cloud_2D,axis=0);
#         mean_x = np.mean(cloud_2D[:,0])
#         mean_y = np.mean(cloud_2D[:,1])
        
        cloud_2D = np.subtract(cloud_2D,mean)
#         cloud_2D[:,0] -= mean_x
#         cloud_2D[:,1] -= mean_y
        # scale the point cloud inside [-1,1]^2
        cloud_2D *= (1.0/14.0)
        cloud_3D = np.concatenate((cloud_2D,np.zeros((k,1))),axis=1)
        return cloud_3D

def incremental_farthest_search(points, k):
    remaining_points = points[:]
    solution_set = []
    solution_set.append(remaining_points.pop(\
                                             random.randint(0, len(remaining_points) - 1)))
    for _ in range(k-1):
        distances = [distance(p, solution_set[0]) for p in remaining_points]
        for i, p in enumerate(remaining_points):
            for j, s in enumerate(solution_set):
                distances[i] = min(distances[i], distance(p, s))
        solution_set.append(remaining_points.pop(distances.index(max(distances))))
    return solution_set


def distance(A,B):
    return sum((A-B)**2)







