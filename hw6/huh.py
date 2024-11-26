import numpy as np
import cv2

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
        [1, 0],
        [1, 1],
        [0, 1]
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