import os
import cv2
import numpy as np
import scipy.linalg 
from scipy.optimize import least_squares

class DirectLinearTransformer():
    def __init__(self):
        # Initialize camera_matrix by 3x4 zeros matrix
        self.target_matrix = np.array([[1, 0, 0, 0], 
                             [0, 1, 0, 0], 
                             [0, 0, 1, 0]], dtype=float)

    def build_measurement_matrix(self, correspondences):
        # TODO #2: Complete the coefficient matrix A, where A * solution = 0 
        # Let the last component = 1 for homogeneous representation of the points
        A = []
        for corr in correspondences:
            # [[X_i, Y_i, Z_i], [x_i, y_i]]
            X_i = corr[0]  # [X_i, Y_i, Z_i]
            x_i = corr[1]  # [x_i, y_i]
            X, Y, Z = X_i
            x, y = x_i
            # Homogeneous coordinates
            X_i_hom = [X, Y, Z, 1]
            zeros = [0, 0, 0, 0]
            # First row
            row1 = X_i_hom + zeros + [-x * X for X in X_i_hom]
            # Second row
            row2 = zeros + X_i_hom + [-y * X for X in X_i_hom]
            A.append(row1)
            A.append(row2)
        A = np.array(A)
        return A

    def update_target_matrix(self, solution):
        # TODO #3: Update self.target matrix from the solution vector
        # 1. Should update the matrix by 'num_rows' and 'num_cols', not specific numbers 3 and 4.
        num_rows, num_cols = self.target_matrix.shape
        self.target_matrix = solution.reshape((num_rows, num_cols))

    def estimation(self, correspondences):
        # Build measurement matrix A
        A = self.build_measurement_matrix(correspondences)
        # Compute SVD of A
        U, S, Vt = np.linalg.svd(A)
        # The solution is the last column of V (last row of Vt)
        solution = Vt[-1, :]
        # Update target matrix from the solution vector
        self.update_target_matrix(solution)
        return self.target_matrix

def calculate_error(correspondences, target_matrix):
    error = 0
    num_corr = len(correspondences)
    for corr in correspondences:
        X_i = corr[0]  # [X_i, Y_i, Z_i]
        x_i = corr[1]  # [x_i, y_i]
        X_i_hom = np.array([X_i[0], X_i[1], X_i[2], 1.0])
        x_i_proj = np.dot(target_matrix, X_i_hom)
        # Normalize so that last component is 1
        x_i_proj = x_i_proj / x_i_proj[2]
        x_i_proj = x_i_proj[:2]  # Only x and y
        x_i = np.array(x_i)
        # Euclidean distance between projected point and observed point
        error += np.linalg.norm(x_i - x_i_proj)
    error /= num_corr
    return error

# def scale_P(P):
#     p3 = P[2, :3]  # p_{31}, p_{32}, p_{33}
#     scale = np.sqrt(np.sum(p3 ** 2))
#     P_scaled = P / scale
#     return P_scaled

def process_file(filename):
    # TODO #1: Load 3D to 2D correspondences from the .txt file in the inputs
    # File format: X Y Z x y on each line, where (X, Y, Z) is the 3D point, (x, y) is the 2D point. 
    # Correspondence format: [[[X1, Y1, Z1], [x1, y1]], [[X2, Y2, Z2], [x2, y2]], ...]
    # Load correspondences
    correspondences = []
    with open(filename, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 5:
                X, Y, Z, x, y = map(float, tokens)
                correspondences.append([[X, Y, Z], [x, y]])
    # Estimate P
    dlt = DirectLinearTransformer()
    P = dlt.estimation(correspondences)
    error = calculate_error(correspondences, P)
    # Scale P
    
    # P_scaled = scale_P(P)
    # error_scaled = calculate_error(correspondences, P_scaled)
    # return P, error, P_scaled, error_scaled, correspondences
    return P, error

# def rq_decomposition(P):
#     # Extract M (first 3 columns of P)
#     M = P[:, :3]
#     # Perform RQ decomposition on M
#     K, R = scipy.linalg.rq(M)
#     # Ensure positive diagonal entries in K
#     T = np.diag(np.sign(np.diag(K)))
#     K = K @ T
#     R = T @ R
#     K = K / K[2, 2]  # Normalize K
#     return K, R

# def compute_t(P, K, R):
#     # Compute translation vector t
#     K_inv = np.linalg.inv(K)
#     t = K_inv @ P[:, 3]
#     return t

def main():
    filenames = ['inputs/corr.txt', 'inputs/corr-subpixel.txt']
    results = []
    for filename in filenames:
        # P, error, P_scaled, error_scaled, correspondences = process_file(filename)
        P, error = process_file(filename)
        
        # RQ decomposition
        # K, R = rq_decomposition(P_scaled)
        # t = compute_t(P_scaled, K, R)
        # Store results
        results.append({
            'filename': filename,
            'P': P,
            'error': error,
            # 'P_scaled': P_scaled,
            # 'error_scaled': error_scaled,
            # 'K': K,
            # 'R': R,
            # 't': t
        })
        # Print results
        print(f"Results for {filename}:")
        print("Estimated P:")
        print(P)
        print(f"Reprojection error: {error}")
        # print("Scaled P:")
        # print(P_scaled)
        # print(f"Reprojection error (scaled): {error_scaled}")
        # print("Calibration matrix K:")
        # print(K)
        # print("Rotation matrix R:")
        # print(R)
        # print("Translation vector t:")
        # print(t)
        print("\n")

if __name__ == "__main__":
    main()