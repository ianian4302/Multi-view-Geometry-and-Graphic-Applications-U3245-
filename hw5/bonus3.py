import os
import cv2
import numpy as np
import scipy.linalg
from scipy.optimize import least_squares

class DirectLinearTransformer():
    def __init__(self):
        # Initialize camera_matrix by 3x4 identity matrix with an extra column of zeros
        self.target_matrix = np.array([[1, 0, 0, 0], 
                                       [0, 1, 0, 0], 
                                       [0, 0, 1, 0]], dtype=float)

    def build_measurement_matrix(self, correspondences):
        # Complete the coefficient matrix A, where A * solution = 0
        A = []
        for corr in correspondences:
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
        # Update self.target matrix from the solution vector
        num_rows, num_cols = self.target_matrix.shape
        self.target_matrix = solution.reshape((num_rows, num_cols))

    def estimation(self, correspondences):
        # Finish estimation pipeline
        A = self.build_measurement_matrix(correspondences)
        # Compute SVD
        U, S, Vt = np.linalg.svd(A)
        # Solution is the last column of V (or last row of Vt)
        solution = Vt[-1, :]
        # Update target matrix
        self.update_target_matrix(solution)
        return self.target_matrix

def calculate_error(correspondences, target_matrix):
    # Calculate geometric distance (reprojection error)
    error = 0
    num_corr = len(correspondences)
    for corr in correspondences:
        X_i = corr[0]
        x_i = corr[1]
        X_i_hom = np.array([X_i[0], X_i[1], X_i[2], 1])
        x_proj_hom = np.dot(target_matrix, X_i_hom)
        x_proj_hom /= x_proj_hom[2]  # Normalize
        x_proj = x_proj_hom[:2]
        x_i = np.array(x_i)
        error += np.linalg.norm(x_proj - x_i) ** 0.5
    error /= num_corr
    return error

def scale_P(P):
    # Scale P such that p_{31}^2 + p_{32}^2 + p_{33}^2 = 1
    p3 = P[2, :3]
    scale_factor = np.sqrt(np.sum(p3 ** 2))
    P_scaled = P / scale_factor
    return P_scaled

def read_correspondences(filename):
    # Load correspondences from the file
    correspondences = []
    with open(filename, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 5:
                X, Y, Z, x, y = map(float, tokens)
                correspondences.append([[X, Y, Z], [x, y]])
    return correspondences

def perform_restricted_camera_estimation(correspondences, initial_params):
    def residuals(params, correspondences):
        alpha = params[0]
        x0 = params[1]
        y0 = params[2]
        rvec = params[3:6]
        tvec = params[6:9]
        # Reconstruct K
        K = np.array([[alpha, 0, x0],
                      [0, alpha, y0],
                      [0, 0, 1]])
        # Reconstruct R
        R, _ = cv2.Rodrigues(rvec)
        # Reconstruct P = K [R | t]
        Rt = np.hstack((R, tvec.reshape(3,1)))
        P = K @ Rt
        residuals = []
        for corr in correspondences:
            X_i = corr[0]
            x_i = corr[1]
            X_i_hom = np.array([X_i[0], X_i[1], X_i[2], 1])
            x_proj_hom = np.dot(P, X_i_hom)
            x_proj_hom /= x_proj_hom[2]
            x_proj = x_proj_hom[:2]
            x_i = np.array(x_i)
            residuals.extend(x_proj - x_i)
        return residuals

    # Pack initial parameters
    params0 = np.hstack([
        initial_params['alpha'],
        initial_params['x0'],
        initial_params['y0'],
        initial_params['rvec'],
        initial_params['tvec']
    ])
    # Perform optimization
    result = least_squares(residuals, params0, args=(correspondences,))
    # Unpack optimized parameters
    params_optimized = result.x
    alpha_opt = params_optimized[0]
    x0_opt = params_optimized[1]
    y0_opt = params_optimized[2]
    rvec_opt = params_optimized[3:6]
    tvec_opt = params_optimized[6:9]
    return {
        'alpha': alpha_opt,
        'x0': x0_opt,
        'y0': y0_opt,
        'rvec': rvec_opt,
        'tvec': tvec_opt,
        'residuals': result.fun
    }

def perform_bonus_3(correspondences, K_init, R_init, t_init):
    # Initialize parameters
    alpha_init = (K_init[0, 0] + K_init[1, 1]) / 2
    x0_init = K_init[0, 2]
    y0_init = K_init[1, 2]
    rvec_init, _ = cv2.Rodrigues(R_init)
    tvec_init = t_init
    initial_params = {
        'alpha': alpha_init,
        'x0': x0_init,
        'y0': y0_init,
        'rvec': rvec_init.flatten(),
        'tvec': tvec_init
    }
    # Report initial parameters
    print("Initial parameters:")
    print(f"alpha: {alpha_init}")
    print(f"x0: {x0_init}")
    print(f"y0: {y0_init}")
    print(f"rvec: {rvec_init.flatten()}")
    print(f"tvec: {tvec_init}")
    # Perform optimization
    optimized_params = perform_restricted_camera_estimation(correspondences, initial_params)
    # Report optimized parameters
    print("Optimized parameters:")
    print(f"alpha: {optimized_params['alpha']}")
    print(f"x0: {optimized_params['x0']}")
    print(f"y0: {optimized_params['y0']}")
    print(f"rvec: {optimized_params['rvec']}")
    print(f"tvec: {optimized_params['tvec']}")
    return optimized_params

def main():
    filenames = ['inputs/corr.txt', 'inputs/corr-subpixel.txt']
    results = []
    for filename in filenames:
        correspondences = read_correspondences(filename)
        dlt = DirectLinearTransformer()
        P = dlt.estimation(correspondences)
        error = calculate_error(correspondences, P)
        P_scaled = scale_P(P)
        error_scaled = calculate_error(correspondences, P_scaled)
        results.append({
            'filename': filename,
            'P': P,
            'error': error,
            'P_scaled': P_scaled,
            'error_scaled': error_scaled,
            'correspondences': correspondences
        })
        print(f"Results for {filename}:")
        print("Estimated P:")
        print(P)
        print(f"Reprojection error: {error}")
        print("Scaled P:")
        print(P_scaled)
        print(f"Reprojection error (scaled): {error_scaled}")
        print("\n")
    # Generate report table
    print("Summary of Results:")
    print("| Filename          | Reprojection Error (P) | Reprojection Error (Scaled P) |")
    print("|-------------------|------------------------|-------------------------------|")
    for res in results:
        print(f"| {res['filename']} | {res['error']:.4f}               | {res['error_scaled']:.4f}                      |")
    # Perform Bonus #1 and #2
    for res in results:
        P_scaled = res['P_scaled']
        K, R = scipy.linalg.rq(P_scaled[:, :3])
        # Ensure positive diagonal entries in K
        T = np.diag(np.sign(np.diag(K)))
        K = K @ T
        R = T @ R
        K /= K[2, 2]  # Normalize K
        res['K'] = K
        res['R'] = R
        # Compute translation vector t
        t = np.linalg.inv(K) @ P_scaled[:, 3]
        res['t'] = t
        print(f"Results for {res['filename']}:")
        print("Calibration matrix K:")
        print(K)
        print("Rotation matrix R:")
        print(R)
        print("Translation vector t:")
        print(t)
        print("\n")
        # Perform Bonus #3
        print(f"Performing restricted camera estimation for {res['filename']}:")
        optimized_params = perform_bonus_3(res['correspondences'], K, R, t)
        res['optimized_params'] = optimized_params

if __name__ == "__main__":
    main()