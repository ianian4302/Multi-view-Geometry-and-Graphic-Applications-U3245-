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
    
    def find_linear_solution_F(self, normalized_points, normalized_points2):
        # Compute the linear solution for the fundamental matrix
        A = np.zeros((normalized_points.shape[0], 9))
        for i in range(normalized_points.shape[0]):
            x, y, k = normalized_points[i]
            x2, y2, k = normalized_points2[i]
            A[i] = [x*x2, x2*y, x2, y2*x, y*y2, y2, x, y, 1]
        U, S, Vt = np.linalg.svd(A)
        F_prime = Vt[-1].reshape(3, 3)
        # Enforce the rank-2 constraint on the fundamental matrix
        U, S, Vt = np.linalg.svd(F_prime)
        S[-1] = 0
        F = np.dot(U, np.dot(np.diag(S), Vt))
        return F_prime, F
        
    def denormalize_F(self, F, T1, T2):
        # Denormalize the fundamental matrix
        return np.dot(T2.T, np.dot(F, T1))
    
    
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
    return np.array(points1), np.array(points2)

# Example usage:
# Load two images for point selection
img1 = cv2.imread('hw7/images/image_01.jpg')
img2 = cv2.imread('hw7/images/image_02.jpg')

img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

eight_points_algorithm = EightPointsAlgorithm()
selected_points1, selected_points2 = select_points_from_images(img1_rgb, img2_rgb)
print("Selected points:", selected_points1, selected_points2)

normalized_points, T = eight_points_algorithm.normalize_points(np.array(selected_points1))
print("Normalized points:", normalized_points)
narmaized_points2, T2 = eight_points_algorithm.normalize_points(np.array(selected_points2))
print("Normalized points2:", narmaized_points2)

F_prime, F = eight_points_algorithm.find_linear_solution_F(normalized_points, narmaized_points2)
print("Fundamental matrix (rank-2 constraint enforced):", F_prime)
print("Fundamental matrix:", F)

F = eight_points_algorithm.denormalize_F(F, T, T2)

print("Determinant of the fundamental matrix (rank-2 constraint enforced):", np.linalg.det(F_prime))
# 0.029826252985013856
print("Determinant of the fundamental matrix:", np.linalg.det(F))

# F -> H, H + pointA -> pointC
# ponitA, pointB
# pointB == pointC 