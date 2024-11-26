import numpy as np
import cv2
import os

def banana(point_x1, point_x2, point_y1, point_y2):
    x1 = point_x1[0]
    y1 = point_x1[1]
    x2 = point_x2[0]
    y2 = point_x2[1]
    a = y2 - y1
    b = x1 - x2
    c = x1 * y2 - x2 * y1

    x1 = point_y1[0]
    y1 = point_y1[1]
    x2 = point_y2[0]
    y2 = point_y2[1]
    a2 = y2 - y1
    b2 = x1 - x2
    c2 = x1 * y2 - x2 * y1

    x = (c * b2 - c2 * b) / (a * b2 - a2 * b)
    y = (c * a2 - c2 * a) / (b * a2 - b2 * a)
    # Return the inhomogeneous coordinate of the vanishing point
    return [x, y, 1]

def compute_calibration_matrix(image_pts_list):
    """
    Computes the camera calibration matrix K from the images of three squares.

    Parameters:
    image_pts_list (list): A list containing three arrays of image points.
                           Each array corresponds to the four corner points
                           of a square in the image.

    Returns:
    K (numpy.ndarray): The 3x3 camera calibration matrix.
    """

    # Define the square points in the plane coordinate system
    square_pts = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=np.float32)

    # Compute homography H for each square
    H_list = []
    for image_pts in image_pts_list:
        H, status = cv2.findHomography(square_pts, image_pts)
        H_list.append(H)

    # Initialize the matrix A for equations A * omega_vector = 0
    A = []
    for H in H_list:
        h1 = H[:, 0]  # First column of H
        h2 = H[:, 1]  # Second column of H

        h1x, h1y, h1z = h1
        h2x, h2y, h2z = h2

        # Equation 1: h1^T * ω * h2 = 0
        eq1 = [
            h1x * h2x,                               # ω11 coefficient
            h1x * h2y + h1y * h2x,                   # ω12 coefficient
            h1x * h2z + h1z * h2x,                   # ω13 coefficient
            h1y * h2y,                               # ω22 coefficient
            h1y * h2z + h1z * h2y,                   # ω23 coefficient
            h1z * h2z                                # ω33 coefficient
        ]
        A.append(eq1)

        # Equation 2: h1^T * ω * h1 - h2^T * ω * h2 = 0
        eq_h1 = [
            h1x * h1x,
            2 * h1x * h1y,
            2 * h1x * h1z,
            h1y * h1y,
            2 * h1y * h1z,
            h1z * h1z
        ]

        eq_h2 = [
            h2x * h2x,
            2 * h2x * h2y,
            2 * h2x * h2z,
            h2y * h2y,
            2 * h2y * h2z,
            h2z * h2z
        ]

        eq2 = [eq_h1[i] - eq_h2[i] for i in range(6)]
        A.append(eq2)

    A = np.array(A)

    # Solve the homogeneous system A * omega_vector = 0 using SVD
    U, S, Vt = np.linalg.svd(A)
    omega_vector = Vt[-1, :]  # Solution corresponds to the smallest singular value

    # Reconstruct the conic ω from omega_vector
    ω11, ω12, ω13, ω22, ω23, ω33 = omega_vector
    omega = np.array([
        [ω11, ω12, ω13],
        [ω12, ω22, ω23],
        [ω13, ω23, ω33]
    ])

    # Compute the inverse of ω
    try:
        omega_inv = np.linalg.inv(omega)
    except np.linalg.LinAlgError:
        print("Inverse of ω cannot be computed. ω might be singular.")
        return None

    # Compute the calibration matrix K from ω_inv = K * K^T
    # Cholesky decomposition
    try:
        K = np.linalg.cholesky(omega_inv).T  # Transpose to get upper-triangular matrix
        # Normalize K so that K[2,2] = 1
        K = K / K[2, 2]
    except np.linalg.LinAlgError:
        print("Cholesky decomposition failed. ω_inv may not be positive-definite.")
        return None

    return K

# Example usage:
# Assuming image_pts_list contains three arrays of image points for the squares
# image_pts_list = [image_pts1, image_pts2, image_pts3]
# K = compute_calibration_matrix(image_pts_list)
# print("Calibration matrix K:\n", K)

#main
image_pts1 = np.array([
    [1129, 1092],
    [2024, 1029],
    [1229, 2105],
    [2049, 1899],
], dtype=np.float32)

# 2449, 1054, 3212, 1154, 2418, 1899, 3143, 2174
image_pts2 = np.array([
    [2449, 1054],
    [3212, 1154],
    [2418, 1899],
    [3143, 2174],
], dtype=np.float32)

# 2230, 2055, 2931, 2374, 1398, 2349, 2036, 2818
image_pts3 = np.array([
    [2230, 2055],
    [2931, 2374],
    [1398, 2349],
    [2036, 2818],
], dtype=np.float32)

image_pts_list = [image_pts1, image_pts2, image_pts3]

K = compute_calibration_matrix(image_pts_list)
print("Calibration matrix K:\n", K)

# a = [251, 1466], b = [715, 1386], c = [270, 1661], d = [843, 1530]
# 計算通過a, b點的線方程式
vanishing_point1 = banana([251, 1466], [715, 1386], [270, 1661], [843, 1530])
print("Vanishing point 1:", vanishing_point1)
vanishing_point2 = banana([251, 1466], [270, 1661], [715, 1386], [843, 1530])
print("Vanishing point 2:", vanishing_point2)

# Determine the line at infinity l' on the image by connecting two vanishing point
vanish_point_1 = vanishing_point1[:2]
vanish_point_2 = vanishing_point2[:2]
l_1 = vanish_point_1[1] - vanish_point_2[1]
l_2 = vanish_point_2[0] - vanish_point_1[0]
l_3 = vanish_point_1[0] * vanish_point_2[1] - vanish_point_2[0] * vanish_point_1[1]
l = [l_1, l_2, l_3]
print("Line at infinity l':", l)

# assume x1 = 300, find y1
x1 = 300
y1 = -(l_3 - l_1 * x1) / l_2
print("y1:", y1)

# assume x2 = 2000, find y2
x2 = 2000
y2 = -(l_3 - l_1 * x2) / l_2
print("y2:", y2)

# caluculate ground plane orientation, orientation_n 垂直於 l 的平面
orientation_n = [l[0], l[1], -l[0] * x1 - l[1] * y1]
print("Ground plane orientation:", orientation_n)

# Calculate the normal vector n from the line at infinity l
n = np.dot(K.T, l)

# Normalize the normal vector
n = n / np.linalg.norm(n)

# Assuming you want to find a rotation matrix R that aligns the z-axis with n
# Create a rotation matrix that aligns the camera's z-axis with the normal vector n
def rotation_matrix_from_normal(n):
    z = n / np.linalg.norm(n)
    x = np.cross(z, np.array([0, 0, 1]))  # Assuming the camera's original z-axis is [0, 0, 1]
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    
    R = np.array([x, y, z]).T
    return R

R = rotation_matrix_from_normal(n)
print("Rotation matrix R:\n", R)

# Step 2: Compute the homography H
# H can be computed using the calibration matrix K and the rotation matrix R
def compute_homography(K, R):
    # Create a 3x4 matrix [R | t] where t is the translation vector
    # Assuming t = [0, 0, 0] for simplicity
    t = np.array([[0], [0], [0]])
    RT = np.hstack((R, t))
    
    # Homography H = K * RT
    H = K @ RT
    return H

H = compute_homography(K, R)
# print("Homography H:\n", H)

# Reshape h into 3x3 matrix H 取左邊三行
H = H[:, :3]
print("Homography H:\n", H)

# Load the original image
original_image = cv2.imread('assests/167859.png')  # 確保這裡的路徑正確

# 獲取原始圖像的寬度和高度
height, width = original_image.shape[:2]  # 獲取圖像的高度和寬度

# 使用 cv2.warpPerspective 進行透視變換
undistorted_image = cv2.warpPerspective(original_image, H, (width, height))

# Dump the undistorted image
# Create output folder if not exist
folder_out = "results"
if not os.path.exists(folder_out):
    os.makedirs(folder_out)

# Write image to the folder
cv2.imwrite(folder_out + '/undistorted.png', undistorted_image)
# cv2.imwrite(folder_out + '/drawed.png', draw_image)

# 顯示或保存變換後的圖像
cv2.imshow('Undistorted Image', undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# P = K [R | t]