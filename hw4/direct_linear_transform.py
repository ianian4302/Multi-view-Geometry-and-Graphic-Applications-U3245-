import os
import cv2
import numpy as np
from matplotlib import image 
from matplotlib import pyplot as plt 


def direct_linear_transform(correspondences):
    H = np.eye(3)  # Initialize homography by 3x3 identity matrix

    # TODO #2: Complete the coefficient matrix A, where Ah = 0
    # Note: h is the solution vector of homography, please see slides.
    # 1. The size of the coefficient matrix A should be 2n x 9, where n is the number of correspondences.
    # 2. Each correspondence contributes two rows to A, please see slides.
    # 3. Report the size of matrix A and the correspondence
    # 4. Report at least two rows of a correspondence to explain that you are correct to write the correspondence matrix
    # Hint: Assume w = 1 for the homogeneous representation

    n = len(correspondences)
    A = np.zeros((2 * n, 9))

    for i, ((x, y), (x_prime, y_prime)) in enumerate(correspondences):
        # Construct the two rows for this correspondence
        A[2 * i] = [x, y, 1, 0, 0, 0, -x * x_prime, -y * x_prime, -x_prime]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -x * y_prime, -y * y_prime, -y_prime]

        # Report the size of A and the correspondence
        if i < 2:
            print(f"Correspondence {i + 1}:")
            print(f"Point in reference image: ({x}, {y})")
            print(f"Point in distorted image: ({x_prime}, {y_prime})")
            print(f"Row {2 * i} of A: {A[2 * i]}")
            print(f"Row {2 * i + 1} of A: {A[2 * i + 1]}")

    print(f"Size of matrix A: {A.shape}")

    # TODO #3: Obtain the svd of A using np.linalg.svd (use default options)
    # Note: Report the outputs of np.linalg.svd

    U, S, Vt = np.linalg.svd(A)
    print("Singular values of A:")
    print(S)

    # TODO #4: Obtain the unit singular vector with least singular value. Please see the slides and np.linalg.svd doc.
    # Note: Report the unit singular vector.

    h = Vt[-1]
    print("Unit singular vector h (corresponding to smallest singular value):")
    print(h)

    # TODO #5: h is the unit singular vector. Update homography H by h (please see slides).
    # Note:
    # 1. Report H.
    # 2. Report the geometric distance (error in one image) for solved H (please see slides)
    # 3. The reported geometric distance should be averaged by number of correspondences.
    # 4. Before compute the error, you should scale the result of x'=Hx so that the third component of x' = 1

    # Reshape h into 3x3 matrix H
    H = h.reshape((3, 3))
    print("Computed homography H:")
    print(H)

    # Compute geometric distance (reprojection error)
    total_error = 0
    for i, ((x, y), (x_prime, y_prime)) in enumerate(correspondences):
        x_ref = np.array([x, y, 1])
        x_proj = H @ x_ref  # x' = H x
        x_proj = x_proj / x_proj[2]  # Normalize so that x_proj[2] = 1

        error = np.sqrt((x_proj[0] - x_prime) ** 2 + (x_proj[1] - y_prime) ** 2)
        total_error += error

        print(
            f"Correspondence {i + 1}: Projected point ({x_proj[0]}, {x_proj[1]}), Original point ({x_prime}, {y_prime}), Error: {error}"
        )

    average_error = total_error / n
    print(f"Average geometric distance (reprojection error): {average_error}")

    return H


# Load test images
reference_image = cv2.imread('images/reference.png')
draw_image = cv2.imread('images/reference.png')
distorted_image = cv2.imread('images/tilted.png')

# Solve homography H by direct linear transform.
# x' = Hx, where x denotes a point in the reference image, x' denotes a point in the distorted image

# TODO #1: assign correspondences:
# The format: [[[x1, y1], [x1', y1']], [[x2, y2], [x2', y2']], ...] (the former is on reference image)
# Note:
# 1. At least select 8 correspondences.
# 2. Draw selected points on the images and attach them in your report

correspondences = [
    [[528, 41], [540, 41]],
    [[715, 36], [715, 35]],
    [[528, 351], [528, 350]],
    [[710, 346], [710, 345]],
    [[276,404], [276, 404]],
    [[576,392], [576, 392]],
    [[282, 575], [282, 575]],
    [[572, 569], [572, 569]]
]
for i, ((x, y), (x_prime, y_prime)) in enumerate(correspondences):
    plt.plot(x ,y ,marker='v' ,color="red" ) 
plt.imshow(draw_image) 
plt.show() 
H = direct_linear_transform(correspondences)

# Generate undistorted image by the solved homography
image_h, image_w = reference_image.shape[0], reference_image.shape[1]
undistorted_image = np.zeros([image_h, image_w, 3], dtype=np.uint8)
for x in range(image_w):
    for y in range(image_h):
        reference_point = np.array([x, y, 1])
        # TODO #6: compute corresponding point on the distorted image
        # Note: you should scale the corresponding point so that its third component = 1
        distorted_point = H @ reference_point
        distorted_point = distorted_point / distorted_point[2]

        # Assign color of corresponding point in the undistorted image
        x2, y2 = int(round(distorted_point[0])), int(round(distorted_point[1]))
        if 0 <= x2 < image_w and 0 <= y2 < image_h:
            undistorted_image[y, x, :] = distorted_image[y2, x2, :]

# Bonus: Use image difference between reference/distorted and reference/undistorted and describe the effect of alignment in the report.

# Dump the undistorted image
# Create output folder if not exist
folder_out = "results"
if not os.path.exists(folder_out):
    os.makedirs(folder_out)
# Write image to the folder
cv2.imwrite(folder_out + '/undistorted.png', undistorted_image)
cv2.imwrite(folder_out + '/drawed.png', draw_image)