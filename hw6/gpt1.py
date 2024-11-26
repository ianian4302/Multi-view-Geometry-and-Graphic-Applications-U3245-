import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.optimize import least_squares

# ------------------------------
# Step 1: Camera Calibration via Metric Planes
# ------------------------------

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
# image_pts1 = np.array([
#     [1129, 1092],
#     [2024, 1029],
#     [1229, 2105],
#     [2049, 1899],
# ], dtype=np.float32)

# # 2449, 1054, 3212, 1154, 2418, 1899, 3143, 2174
# image_pts2 = np.array([
#     [2449, 1054],
#     [3212, 1154],
#     [2418, 1899],
#     [3143, 2174],
# ], dtype=np.float32)

# # 2230, 2055, 2931, 2374, 1398, 2349, 2036, 2818
# image_pts3 = np.array([
#     [2230, 2055],
#     [2931, 2374],
#     [1398, 2349],
#     [2036, 2818],
# ], dtype=np.float32)

image_pts1 = np.array([
    [472, 491],
    [983, 436],
    [571, 970],
    [1012, 865],
], dtype=np.float32)

image_pts2 = np.array([
    [1287,452 ],
    [1802, 484],
    [1277, 881],
    [1725,938],
], dtype=np.float32)

image_pts3 = np.array([
    [788, 1162],
    [1309,1070],
    [852, 1581],
    [1552, 1405],
], dtype=np.float32)

image_pts_list = [image_pts1, image_pts2, image_pts3]

K = compute_calibration_matrix(image_pts_list)
print("Calibration matrix K:\n", K)

# ------------------------------
# Step 2: Estimate Ground Plane Orientation
# ------------------------------

# Load the image of the hall
hall_img = cv2.imread('assests/167856.png')  # Replace with your image path
if hall_img is None:
    print("Hall image not found. Please check the path.")
    exit()
hall_img_rgb = cv2.cvtColor(hall_img, cv2.COLOR_BGR2RGB)

# Display the image and collect points
plt.figure(figsize=(10, 8))
plt.imshow(hall_img_rgb)
plt.title('Click points along two sets of parallel lines (press Enter when done)')
plt.axis('off')

# Collect points for the first set of parallel lines
print("Select points along the first set of parallel lines (at least two lines).")
lines1 = plt.ginput(n=-1, timeout=0)
plt.show()

# Collect points for the second set of parallel lines
plt.figure(figsize=(10, 8))
plt.imshow(hall_img_rgb)
plt.title('Click points along the second set of parallel lines (press Enter when done)')
plt.axis('off')
print("Select points along the second set of parallel lines (at least two lines).")
lines2 = plt.ginput(n=-1, timeout=0)
plt.show()

# Organize points into lines (pairs of points)
def organize_points(points):
    return [(points[i], points[i+1]) for i in range(0, len(points)-1, 2)]

line_pairs1 = organize_points(lines1)
line_pairs2 = organize_points(lines2)

# Function to compute line coefficients (ax + by + c = 0)
def compute_line(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    line = np.cross([x1, y1, 1], [x2, y2, 1])
    return line / np.linalg.norm(line[:2])

# Compute lines and vanishing points
line_coeffs1 = [compute_line(p1, p2) for p1, p2 in line_pairs1]
line_coeffs2 = [compute_line(p1, p2) for p1, p2 in line_pairs2]

# Compute vanishing points
def compute_intersection(l1, l2):
    vp = np.cross(l1, l2)
    vp = vp / vp[2]
    return vp

if len(line_coeffs1) >= 2 and len(line_coeffs2) >= 2:
    vp1 = compute_intersection(line_coeffs1[0], line_coeffs1[1])
    vp2 = compute_intersection(line_coeffs2[0], line_coeffs2[1])
else:
    print("Not enough lines were selected. Please select at least two lines for each set.")
    exit()

print("\nVanishing Point 1:", vp1)
print("Vanishing Point 2:", vp2)

# Vanishing line (line at infinity)
vanishing_line = np.cross(vp1, vp2)
vanishing_line = vanishing_line / np.linalg.norm(vanishing_line[:2])

print("\nVanishing Line:", vanishing_line)

# Calculate the plane normal in camera coordinates
l = vanishing_line
n = np.linalg.inv(K.T) @ l
n = n / np.linalg.norm(n)

print("\nGround Plane Normal in Camera Coordinates:", n)

# Visualize the vanishing points on the image
plt.figure(figsize=(10, 8))
plt.imshow(hall_img_rgb)
plt.plot(vp1[0], vp1[1], 'ro', label='Vanishing Point 1')
plt.plot(vp2[0], vp2[1], 'bo', label='Vanishing Point 2')
plt.legend()
plt.title('Image with Vanishing Points')
plt.axis('off')
plt.show()

# ------------------------------
# Step 3: Rectify the Image
# ------------------------------

# Function to compute rotation matrix that aligns vec1 to vec2
def rotation_matrix_from_vectors(vec1, vec2):
    """Compute the rotation matrix that aligns vec1 to vec2."""
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.identity(3)  # No rotation needed
    kmat = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    rotation_mat = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))
    return rotation_mat

# Our vectors
n = n.reshape(3)
ez = np.array([0, 0, 1])

# Compute rotation matrix R to align ground plane normal to camera's z-axis
R = rotation_matrix_from_vectors(n, ez)

print("\nRotation Matrix R:")
print(R)

# Compute the homography H
H = K @ R @ np.linalg.inv(K)
H /= H[2, 2]  # Normalize

print("\nHomography Matrix H:")
print(H)

# [[ 1.00000000e+00 -1.22222166e-10  2.22656100e-04]
#  [ 2.31530432e-12  1.00000000e+00  8.82065455e-04]
#  [ 5.15837918e-12 -1.50486600e-10  1.00000000e+00]]
H_1 = [
    [ 1.00000000e+00, -1.22222166e-4,  2.22656100e-02],
    [ 2.31530432e-5,  1.00000000e+00,  8.82065455e-02],
    [ 5.15837918e-5, -1.50486600e-4,  1.00000000e+00]
]
H_2 = np.array(H_1)
# print(H_2)

# Rectify the image using the homography
img_height, img_width = hall_img.shape[:2]
rectified_img = cv2.warpPerspective(hall_img, H_2, (img_width, img_height))

# Display the original and rectified images
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.imshow(hall_img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(rectified_img, cv2.COLOR_BGR2RGB))
plt.title('Rectified Image')
plt.axis('off')

plt.show()