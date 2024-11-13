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
            # -1 * X_i_hom
            row1 = zeros + [-1 * X for X in X_i_hom] + [y * X for X in X_i_hom]
            # Second row
            row2 = X_i_hom + zeros + [-x * X for X in X_i_hom]
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
        # TODO #4: Finish estimation pipeline by the following tools
        # 1. self.build_measurement_matrix
        # 2. np.linalg.svd
        # 3. self.update_target_matrix
        
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
    # TODO #5: Calculate geometric distance (reprojection error, please see slides) 
    # 1. The error should be averaged by number of correspondences. 
    # 2. You should normalized the projected point so that the last component of point is 1
    
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
        error += np.linalg.norm(x_i - x_i_proj) ** 0.5
    error /= num_corr
    return error

def scale_P(P):
    p3 = P[2, :3]  # p_{31}, p_{32}, p_{33}
    scale = np.sqrt(np.sum(p3 ** 2))
    P_scaled = P / scale
    return P_scaled

def process_file(filename):
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
    P_scaled = scale_P(P)
    error_scaled = calculate_error(correspondences, P_scaled)
    return P, error, P_scaled, error_scaled, correspondences

def rq_decomposition(P):
    # Extract M (first 3 columns of P)
    M = P[:, :3]
    # Perform RQ decomposition on M
    K, R = scipy.linalg.rq(M)
    # Ensure positive diagonal entries in K
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R
    K = K / K[2, 2]  # Normalize K
    return K, R

def compute_t(P, K, R):
    # Compute translation vector t
    K_inv = np.linalg.inv(K)
    t = K_inv @ P[:, 3]
    return t

def main():
    filenames = ['inputs\corr.txt', 'inputs\corr-subpixel.txt']
    results = []
    for filename in filenames:
        P, error, P_scaled, error_scaled, correspondences = process_file(filename)
        # RQ decomposition
        K, R = rq_decomposition(P_scaled)
        t = compute_t(P_scaled, K, R)
        # Store results
        results.append({
            'filename': filename,
            'P': P,
            'error': error,
            'P_scaled': P_scaled,
            'error_scaled': error_scaled,
            'K': K,
            'R': R,
            't': t
        })
        # Print results
        print(f"Results for {filename}:")
        print("Estimated P:")
        print(P)
        print(f"Reprojection error: {error}")
        print("Scaled P:")
        print(P_scaled)
        print(f"Reprojection error (scaled): {error_scaled}")
        print("Calibration matrix K:")
        print(K)
        print("Rotation matrix R:")
        print(R)
        print("Translation vector t:")
        print(t)
        print("\n")

if __name__ == "__main__":
    main()
    
# TODO: Reports for #1-#6
# 1. Test on two sets of inputs: corr.txt and corr-subpixel.txt. Report following results:
# - Estimated camera matrix P and its scaled results
# - Reprojection errors for P and scaled P
# Note: please well organized all the results in a single table in the doc.
# Bonus #1: Use scipy.linalg.rq to perform RQ decomposition on the 'scaled' camera matrix to get the values of calibration matrix K and R
# 1. Report K, R, K@R for both scaled camera matrices solved by using corr.tex and corr-subpixel.txt in the doc.
# 2. Should ensure the diagonal entries of K be positive.
# 3. Provide implementation here.

# Bonus #2: Solve translation t to finalize the decomposition of camera matrix P=K[R|t]
# 1. Share your ideas in the doc.
# 2. Provide implementation here.

# Bonus #3: Restricted camera estimation (please see slides for the following notations)
# 1. Initialize alpha = alpha_x = alpha_y in calibration matrix K by averaging entries of P
# 2. Use the decomposed matrix K to directly assign x0, y0
# 3. Convert the decomposed rotation matrix R to 3-d angle axis vector by cv2.rodrigues and use the vector to represent rotation parameters.
# 4. Use the decomposed translation t to represent translation parameters
# 5. Report the initial 9 variable parameters (alpha, x0, y0, rotation, translation) in the doc.
# 6. Use scipy.optimize.least_squares for refining the 9 variable parameters
# 7. Report the final 9 variable parameters in the doc.
# 8. Share the formulations in the doc. and provide implementation here.