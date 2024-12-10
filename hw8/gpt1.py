import numpy as np

def apply_homography(H, points):
    """
    Apply homography H to a set of points (N x 2 or N x 3).
    Points should be in the format Nx2 or Nx3 (with last coordinate=1).
    Returns transformed points normalized so last coordinate = 1.
    """
    # Ensure homogeneous form
    if points.shape[1] == 2:
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    else:
        points_h = points.copy()

    transformed = (H @ points_h.T).T
    # Normalize
    transformed /= transformed[:, [2]]
    return transformed

def direct_linear_transform(correspondences):
    # This is your existing DLT code, as provided in your snippet.
    H = np.eye(3)

    n = len(correspondences)
    A = np.zeros((2 * n, 9))

    for i, ((x, y), (x_prime, y_prime)) in enumerate(correspondences):
        A[2 * i] = [x, y, 1, 0, 0, 0, -x * x_prime, -y * x_prime, -x_prime]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -x * y_prime, -y * y_prime, -y_prime]

    # SVD
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H = h.reshape((3, 3))

    # Normalize H so that H[2,2] = 1 if possible
    if H[2,2] != 0:
        H = H / H[2,2]

    return H

# Suppose we have:
# correspondences = [( (x1,y1), (x1',y1') ), ..., ((xn,yn),(xn',yn')) ]
# and we have computed a homography H_prime.

# Example:
correspondences = [
    ((100, 50), (200, 80)),
    ((120, 60), (220, 90)),
    ((130, 70), (230, 100)),
    ((140, 75), (240, 105))
]  # Just an example; use your actual correspondences

H_prime = np.array([
    [ 1.2, 0.02,  10],
    [ 0.01, 1.1,  20],
    [ 0.001, 0.002, 1]
])

# Step 1: Transform the second set of points by H'
points1 = np.array([c[0] for c in correspondences])  # (x_i, y_i)
points2 = np.array([c[1] for c in correspondences])  # (x_i', y_i')

points2_transformed = apply_homography(H_prime, points2)  # x''_i = H' x_i'

# Step 2: Now solve for H using (x_i, x''_i)
new_correspondences = list(zip(points1, points2_transformed[:, :2]))  
H = direct_linear_transform(new_correspondences)

print("Computed homography H that minimizes sum of distances d(Hx_i, H'x_i'):")
print(H)

# You can verify the quality by measuring the distances again:
total_error = 0
for i, (p1, p2) in enumerate(zip(points1, points2)):
    x_proj = apply_homography(H, p1.reshape(1,2))[0]   # Hx_i
    x_target = apply_homography(H_prime, p2.reshape(1,2))[0]  # H'x_i'
    error = np.linalg.norm(x_proj[:2] - x_target[:2])
    total_error += error
    print(f"Correspondence {i+1}: error={error}")

average_error = total_error / len(points1)
print("Average error after finding H:", average_error)