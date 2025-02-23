import cv2
import numpy as np
import os
from scipy.ndimage import distance_transform_edt
import heapq
import time



def fast_marching_method(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Canny is O(n) but only gives edges
    edges = cv2.Canny(image, threshold1=100, threshold2=200)
    # find contours is O(n) and gives all contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary_indices = np.concatenate(np.vstack(contours)  ) 
    height, width = image.shape
    dist = np.full((height, width), np.inf) # distance array initialized to infinity
    heap = []     # heap for Fast Marching
    for x, y in boundary_indices:
        dist[y, x] = 0
        heapq.heappush(heap, (0, (y, x)))  # Push (distance, (y, x)) into the heap
    
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while heap:
        current_dist, (y, x) = heapq.heappop(heap) # Pop the pixel with the smallest distance
        # Iterate over all the 4 neighbors (up, down, left, right)
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            # Ensure the neighbor is within bounds
            if 0 <= ny < height and 0 <= nx < width:
                # Update the neighbor's distance if it is not processed (np.inf)
                    new_dist = current_dist + 1  # Simplified update rule (step of 1)
                    if new_dist < dist[ny, nx] or np.isinf(dist[ny, nx]):
                        dist[ny, nx] = new_dist
                        heapq.heappush(heap, (new_dist, (ny, nx)))
    return dist


if __name__ == '__main__':
    image = cv2.imread('test_images/image.png')
    start = time.time()
    distance_field = fast_marching_method(image)
    print("Execution time in ms:", (time.time() - start) * 1000)
    start = time.time()

    nan_values = np.isnan(distance_field)
    inf_values = np.isinf(distance_field)

    if np.any(nan_values):
        print("NaN values found at:")
        print(np.column_stack(np.where(nan_values)))
    if np.any(inf_values):
        print("Inf values found at:")
        print(np.column_stack(np.where(inf_values)))

    distance_field = np.nan_to_num(distance_field, nan=0, posinf=np.inf, neginf=0)

    distance_field_normalized = cv2.normalize(distance_field, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    distance_colormap = cv2.applyColorMap(distance_field_normalized, cv2.COLORMAP_JET)

    cv2.imshow('Original Image', image)
    cv2.imshow('Distance Field Heatmap', distance_colormap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()