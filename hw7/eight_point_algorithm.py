import numpy as np
import matplotlib.pyplot as plt
import cv2

class EightPointsAlgorithm:
    def __init__(self):
        self.F = None
        
    def normalize_points(self, points):
        # Normalize the points
        # 1. Translate the points so that their centroid is at the origin
        mean = np.mean(points, axis=0)
        # 2. Scale the points so that the average distance from the origin is sqrt(2)
        std = np.std(points)
        # 3. Combine the two transformations into a single transformation matrix
        T = np.array([[1/std, 0, -mean[0]/std],
                      [0, 1/std, -mean[1]/std],
                      [0, 0, 1]])
        return np.dot(T, np.vstack((points.T, np.ones(points.shape[0])))).T, T
    
def select_points_from_images(img1, img2):
    plt.figure(figsize=(12, 6))

    # Show the first image
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title('Select a point in the left image')
    plt.axis('off')
    points1 = plt.ginput(n=-1, timeout=0)  # Allow user to select points

    # Show the second image
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title('Select a point in the right image')
    plt.axis('off')
    points2 = plt.ginput(n=-1, timeout=0)  # Allow user to select points

    plt.close()  # Close the plot after selection
    corresponeces = np.array(list(zip(points1, points2)))
    return corresponeces

# Example usage:
# Load two images for point selection
img1 = cv2.imread('hw7/images/image_01.jpg')
img2 = cv2.imread('hw7/images/image_02.jpg')

img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

eight_points_algorithm = EightPointsAlgorithm()
selected_points = select_points_from_images(img1_rgb, img2_rgb)
print("Selected points:", selected_points)

