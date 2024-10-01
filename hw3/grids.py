class Grids():
    def __init__(self):
        # Settings for the grids
        self.num_rows = 9
        self.num_cols = 12
        self.intv_mm = 30 # Interval between grid points in millimeter
        self.dist_mm = 200 # Distance to origin in millimeter

    def generate_points(self):
        grids = [] 
        # Compute center of the grids.
        center_x = self.num_cols / 2 * self.intv_mm
        center_y = self.num_rows / 2 * self.intv_mm
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                # Calculate grid point coordinates with center as the origin.
                grids.append([c * self.intv_mm - center_x, r * self.intv_mm - center_y, self.dist_mm])
        return grids

    def generate_line_indices(self):
        line_indices = []
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                if c < self.num_cols-1:
                    line_indices.append([r * self.num_cols + c, r * self.num_cols + (c + 1)])
                if r < self.num_rows-1:
                    line_indices.append([r * self.num_cols + c, (r + 1) * self.num_cols + c])
        return line_indices

