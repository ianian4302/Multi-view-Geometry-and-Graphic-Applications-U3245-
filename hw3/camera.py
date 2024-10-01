import numpy as np

class Camera():
    def __init__(self, fov, image_w, image_h):
        # Projection matrix
        F = fov * np.pi / 180 
        w, h = image_w, image_h
        f = h / (2 * (np.tan(F / 2))) #Focal length
        self.P = np.array([[f, 0, w / 2],
                           [0, f, h / 2],
                           [0, 0, 1]])
        # Store focal length
        self.focal = f

        # Scanned image: raster scanning on the latent image
        self.I = np.zeros([h, w, 3])

        # Store image size
        self.w = w
        self.h = h
        
    def project_to_image_position(self, X):
        X = np.array(X)
        x = self.P @ X
        x /= x[2]
        # Flip Y to fit image direction
        x[1] = self.h - x[1] 
        # Floor to get integer position
        return x[:2].astype(int)
