import numpy as np

class ProjectiveMapping():
    def __init__(self, a0, a1, a2, b0, b1, b2, c1, c2):
        self.params = [a0, a1, a2, b0, b1, b2, c1, c2]

    def generate_motion(self, p):
        '''
        px: x coordinate of the imaged point
        py: y coordinate of the imaged point
        dx: x component of the displacement (motion)
        dy: y component of the displacement (motion)
        '''
        px, py = p[0], p[1]
        a0, a1, a2, b0, b1, b2, c1, c2 = self.params
        dx = (a0 + a1*px + a2*py) / (1 + c1*px + c2*py) - px
        dy = (b0 + b1*px + b2*py) / (1 + c1*px + c2*py) - py
        return [dx, dy]

# theta: in radian
def compute_rotation_x(theta):
    return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

# theta: in radian
def compute_rotation_y(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

# theta: in radian
def compute_rotation_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
