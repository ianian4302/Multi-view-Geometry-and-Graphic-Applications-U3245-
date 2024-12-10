import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.linalg as linalg
    
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

def find_matching_transformation(points1, points2):
    # 使用 OpenCV 計算變換矩陣 H
    H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
    return H

def synthesize_images(img1, img2, H, H_prime):
    # 對圖像進行變形
    warped_img1 = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))
    warped_img2 = cv2.warpPerspective(img2, H_prime, (img2.shape[1], img2.shape[0]))
    return warped_img1, warped_img2

def normalize_points(points):
    mean = np.mean(points, axis=0)
    std = np.std(points)
    T = np.array([[1/std, 0, -mean[0]/std],
                  [0, 1/std, -mean[1]/std],
                  [0, 0, 1]])
    normalized_points = np.dot(T, np.vstack((points.T, np.ones(points.shape[0])))).T
    return normalized_points, T

def compute_fundamental_matrix(points1, points2):
    # 正規化點
    normalized_points1, T1 = normalize_points(points1)
    normalized_points2, T2 = normalize_points(points2)

    # 建立矩陣 A
    A = np.zeros((normalized_points1.shape[0], 9))
    for i in range(normalized_points1.shape[0]):
        x, y, _ = normalized_points1[i]
        x2, y2, _ = normalized_points2[i]
        A[i] = [x * x2, x2 * y, x2, y * x2, y * y2, y2, x, y, 1]

    # 使用 SVD 計算 F'
    U, S, Vt = np.linalg.svd(A)
    F_prime = Vt[-1].reshape(3, 3)

    # 強制執行秩為 2 的約束
    U, S, Vt = np.linalg.svd(F_prime)
    S[-1] = 0  # 將最後一個奇異值設置為 0
    F = np.dot(U, np.dot(np.diag(S), Vt))

    # 返回基本矩陣
    return F, T1, T2

def compute_h_prime(epipole):
    # 確保 epipole 是一維數組
    if epipole.ndim > 1:
        epipole = epipole.flatten()
    # Construct the projective transformation H' to map epipole to infinity
    H_prime = np.array([[1, 0, -epipole[0]],
                        [0, 1, -epipole[1]],
                        [0, 0, 1]])
    return H_prime

# Example usage:
# Load two images for point selection
img1 = cv2.imread('hw7/images/image_01.jpg')
img2 = cv2.imread('hw7/images/image_02.jpg')

img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# 1. Select points from the images
selected_points1, selected_points2 = select_points_from_images(img1_rgb, img2_rgb)
print("Selected points:", selected_points1, selected_points2)

# 2. Compute the fundamental matrix F
points1 = np.array(selected_points1)  # 替換為實際點
points2 = np.array(selected_points2)  # 替換為實際點
F, T1, T2 = compute_fundamental_matrix(points1, points2)
print("Fundamental Matrix:", F)

# Compute the epipole e' in the second image
epipole_e_prime = linalg.null_space(F)
epipole_e_prime = epipole_e_prime.flatten()  # 將其展平為一維數組
print("Epipole e' shape:", epipole_e_prime.shape)
print("Epipole e' value:", epipole_e_prime)

H = find_matching_transformation(selected_points1, selected_points2)
print("Matching transformation H that minimizes the least square distance:", H)

H_prime = compute_h_prime(epipole_e_prime)
print("Projective transformation H' that maps the epipole to infinity:", H_prime)

warped_img1, warped_img2 = synthesize_images(img1_rgb, img2_rgb, H, H_prime)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(warped_img1)
plt.title('Warped Image 1')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(warped_img2)
plt.title('Warped Image 2')
plt.axis('off')

plt.show()

