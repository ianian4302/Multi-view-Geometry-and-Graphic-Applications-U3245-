import os
import cv2
from camera import *
from grids import *
from motion_generator import *

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

def affine_rectification(points):
    rectified_points = [] # Initialize the rectified points
    # TODO #1: Assign four points of first parallel lines on the image 'projectivity', two of each line.
    #      (Hint: you can choose the points from the dump image or print projective points at specific row and column)
    #      (Note: Draw selected points on image 'projectivity' and attach them in your report)
    #   grid 9 * 12
    #70, 77, 94, 101
    #9, 11, 21, 23
    a = 0
    picked_points = []
    picked_points.append(points[9])
    picked_points.append(points[11])
    picked_points.append(points[21])
    picked_points.append(points[23])
    # TODO #2: Compute their vanishing point (the intersection point of lines on 'projectivity'.
    #      (Note: Write inhomogenous coordinate of vanishing point in your report)
    banana1 = banana(picked_points[0], picked_points[1], picked_points[2], picked_points[3])
    
    # TODO #3: Similarly, assign two points of second parallel lines and compute their vanishing point.
    #      (Note: Provide your selected points and inhomgeneous coordinate of second vanishing point in a similar manner)
    banana2 = banana(picked_points[0], picked_points[2], picked_points[1], picked_points[3])

    # TODO #4: Determine the line at infinity l' on the image by connecting two vanishing point
    #      (Note: please report line at infity with 3rd coordinate = 1)
    vanish_point_1 = banana1[:2]
    vanish_point_2 = banana2[:2]
    l_1 = vanish_point_1[1] - vanish_point_2[1]
    l_2 = vanish_point_2[0] - vanish_point_1[0]
    l_3 = vanish_point_1[0] * vanish_point_2[1] - vanish_point_2[0] * vanish_point_1[1]
    l = [l_1, l_2, l_3]

    # TODO #5: Rectify all points by rectified_points = H * points, where H: [1 0 0; 0 1 0; l_1/l_3 l_2/l_3 1], and l'=[l_1; l_2; l_3]
    H = np.array([[1, 0, 0], [0, 1, 0], [l_1/l_3, l_2/l_3, 1]])
    
    for point in points:
        #[x, y, 1]
        homopoint = [point[0], point[1], 1]
        rectified_points.append((H @ np.array(homopoint))[:3])

    #  The operation is in terms of 'homogeneous', and the rectified point should be scaled such that the 
    #       third component is '1' for 'inhomogenous result'). 
    for idx, rectified_point in enumerate(rectified_points):
        rectified_point /= rectified_point[2]
        rectified_point = rectified_point[:2].astype(int)
        rectified_points[idx] = rectified_point
      
    # TODO #6: Return the rectified points instead of points
    return rectified_points

# Create object
grids = Grids()
points = grids.generate_points()

# Create camera
fov, image_w, image_h = 60, 800, 600
camera = Camera(fov, image_w, image_h)

# Points after similarity
# 1. Project the grids without pre-rotation 
similarity_points = []
for point in points:
    similarity_points.append(camera.project_to_image_position(point))

# Points after projective
# 1.Rotate the grids
# Generate rotation along x-axis and z-axis
theta_x_deg = 15 # Rotation angle in degree
theta_y_deg = 15 # Rotation angle in degree
Rx = compute_rotation_x(np.radians(theta_x_deg))
Ry = compute_rotation_y(np.radians(theta_x_deg))
rotated_points = []
for point in points:
    rotated_points.append(Ry @ Rx @ np.array(point))
# 2.Project the grids to the image plane
projective_points = []
for point in rotated_points:
    projective_points.append(camera.project_to_image_position(point))

# Affine rectification for the projected points
affine_points = affine_rectification(projective_points)

# Render the grid image
# Generate line indices to connect the points    
line_indices = grids.generate_line_indices()
color = [0, 255, 0] # Line color
thickness = 3 # Line thickness
# Draw lines
similarity_image = np.zeros([image_h, image_w, 3], dtype=np.uint8)
affine_image = np.zeros([image_h, image_w, 3], dtype=np.uint8)
projective_image = np.zeros([image_h, image_w, 3], dtype=np.uint8)

for pair in line_indices:
    p, q = pair[0], pair[1]
    cv2.line(similarity_image, similarity_points[p], similarity_points[q], color, thickness)
    cv2.line(affine_image, affine_points[p], affine_points[q], color, thickness)
    cv2.line(projective_image, projective_points[p], projective_points[q], color, thickness)

# Dump the projected image
# Create output folder if not exist
folder_out = "results"
if not os.path.exists(folder_out):
    os.makedirs(folder_out)
# Write image to the folder
cv2.imwrite(folder_out+'/similarity.png', similarity_image)
cv2.imwrite(folder_out+'/projectivity.png', projective_image)
cv2.imwrite(folder_out+'/affinity.png', affine_image)

