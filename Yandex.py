import numpy as np
# do not change the code in the block below
# __________start of block__________
class DummyMatch:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx  # index in des1
        self.trainIdx = trainIdx  # index in des2
        self.distance = distance
# __________end of block__________


def match_key_points_numpy(des1: np.ndarray, des2: np.ndarray) -> list:
    dist_matrix = np.sqrt(np.sum((des1[:, np.newaxis] - des2) ** 2, axis=2))
    matches_1_to_2 = np.argmin(dist_matrix, axis=1)
    min_dist_1_to_2 = np.min(dist_matrix, axis=1)
    

    matches_2_to_1 = np.argmin(dist_matrix, axis=0)
    min_dist_2_to_1 = np.min(dist_matrix, axis=0)
    
    mutual_matches = []
    for i in range(des1.shape[0]):
        j = matches_1_to_2[i]
        if matches_2_to_1[j] == i:
            distance = dist_matrix[i, j]
            mutual_matches.append(DummyMatch(i, j, distance))
    
    mutual_matches.sort(key=lambda x: x.distance)
    
    return mutual_matches
