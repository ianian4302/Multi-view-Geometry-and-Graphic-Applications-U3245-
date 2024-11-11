import os
import cv2
import numpy as np
import scipy.linalg

class DirectLinearTransformer():
    def __init__(self):
        # Initialize camera_matrix by 3x3 identity matrix
        self.target_matrix = np.array([[1, 0, 0, 0], 
                             [0, 1, 0, 0], 
                             [0, 0, 1, 0]], dtype=float)

    def build_measurement_matrix(self, correspondences):
        # TODO #2: Complete the coefficient matrix A, where A * solution = 0 
        # Let the last component = 1 for homogeneous representation of the points
        A = []
        A = np.array(A)
        return A

    def update_target_matrix(self, solution):
        # TODO #3: Update self.target matrix from the solution vector
        # 1. Should update the matrix by 'num_rows' and 'num_cols', not specific numbers 3 and 4.
        num_rows, num_cols = self.target_matrix.shape[0], self.target_matrix.shape[1]

    def estimation(self, correspondences):
        # TODO #4: Finish estimation pipeline by the following tools
        # 1. self.build_measurement_matrix
        # 2. np.linalg.svd
        # 3. self.update_target_matrix

        return self.target_matrix

def calculate_error(correspondences, target_matrix):
    # TODO #5: Calculate geometric distance (reprojection error, please see slides) 
    # 1. The error should be averaged by number of correspondences. 
    # 2. You should normalized the projected point so that the last component of point is 1
    error = 0

    return error


# TODO #1: Load 3D to 2D correspondences from the .txt file in the inputs
# File format: X Y Z x y on each line, where (X, Y, Z) is the 3D point, (x, y) is the 2D point. 
# Correspondence format: [[[X1, Y1, Z1], [x1, y1]], [[X2, Y2, Z2], [x2, y2]], ...]
correspondences = []
#25.524054601740488 -225.63045775253644 597.6472088623268 422.191499632132 496.17095709675164

dlt = DirectLinearTransformer()
P = dlt.estimation(correspondences)

error = calculate_error(correspondences, P)

# TODO #6: Scale P such that p_{31}^2 + p_{32}^2 + p_{33}^2 = 1
# Report the scaled P and its error in the doc.

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
