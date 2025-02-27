import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import voronoi

####################################################################################################
# QUADTREE IMPLEMENTATION
####################################################################################################

class Quadtree:
    def __init__(self, boundary, capacity=4):
        """
        Initialize Quadtree.

        Args:
            boundary (tuple): Area boundary.
            capacity (int): Maximum number of points before subdividing.
        """
        self.boundary = boundary  # Area boundary
        self.capacity = capacity  # Maximum number of points before subdividing
        self.points = []  # List of points in the quadtree
        self.divided = False  # Indicative if the quadtree is divided
        self.quadrants = []  # Quadrants

    def Insert(self, point):
        """
        Insert a point into the quadtree.
        
        Args:
            point (tuple): Point to be inserted.
        
        Returns:
            bool: True if the point was inserted, False otherwise.
        """
        if not self.InBoundary(point):
            return False
        
        if len(self.points) < self.capacity:
            self.points.append(point)
            return True

        if not self.divided:
            self.Subdivide()

        # Redistribute points to quadrants
        for p in self.points:
            for quadrant in self.quadrants:
                if quadrant.InBoundary(p):
                    quadrant.Insert(p)
        self.points = []  # Clear points in this node after redistribution

        for quadrant in self.quadrants:
            if quadrant.Insert(point):
                return True
        
        return False

    def InBoundary(self, point):
        """
        Check if a point is within the quadtree boundary.
        
        Args:
            point (tuple): Point to be checked.
        
        Returns:
            bool: True if the point is within the boundary, False otherwise.
        """
        xmin, ymin, xmax, ymax = self.boundary
        x, y = point
        return xmin <= x <= xmax and ymin <= y <= ymax

    def Subdivide(self):
        """
        Subdivide the quadtree into four quadrants.
        
        The boundary is divided into four quadrants:
        - Upper left
        - Upper right
        - Lower left
        - Lower right
        """
        xmin, ymin, xmax, ymax = self.boundary
        mid_x = (xmin + xmax) / 2
        mid_y = (ymin + ymax) / 2

        # Create quadrants
        self.quadrants = [
            Quadtree((xmin, ymin, mid_x, mid_y), self.capacity),  # Upper left quadrant
            Quadtree((mid_x, ymin, xmax, mid_y), self.capacity),  # Upper right quadrant
            Quadtree((xmin, mid_y, mid_x, ymax), self.capacity),  # Lower left quadrant
            Quadtree((mid_x, mid_y, xmax, ymax), self.capacity)   # Lower right quadrant
        ]

        self.divided = True
    
    def Plot(self, ax):
        """
        Plot the quadtree.
        
        Args:
            ax (matplotlib.axes.Axes): Axes object.
        
        Returns:
            None
        """
        xmin, ymin, xmax, ymax = self.boundary
        ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'k-')
        
        if self.divided:
            for quadrant in self.quadrants:
                quadrant.Plot(ax) 
    
    def GetVertices(self):
        """
        Get the vertices of the quadtree.
        
        Returns:
            list: Vertices of the quadtree.
        """
        vertices = set()
        stack = [self]
        while stack:
            node = stack.pop()
            xmin, ymin, xmax, ymax = node.boundary
            vertices.update([(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)])
            if node.divided:
                stack.extend(node.quadrants)
        return list(vertices)  
    
####################################################################################################
# FUNCTIONS TO LOAD DATA AND FILTER DATAFRAME AND PLOT
####################################################################################################
  
def LoadFileToDataframe(file_path="models.xlsx"):
    """
    Load file to pandas dataframe.

    Args:
        file_path (str): File path
    Returns:
        pd.DataFrame: Dataframe
    """

    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Could not load the file: {e}")
        return None
    

def FilterDataframe(df):
    """
    Filter the dataframe to get only the necessary information and find the robot initial point.
    """
    # Remove extra robot coordinates
    df_filtered = df.drop(df[(df["Model Name"] == "rover_argo_NZero::base_link") | 
                                        (df["Model Name"] == "rover_argo_NZero::front_left_wheel_link") | 
                                        (df["Model Name"] == "rover_argo_NZero::front_right_wheel_link") |
                                        (df["Model Name"] == "rover_argo_NZero::rear_right_wheel_link") |
                                        (df["Model Name"] == "rover_argo_NZero::imu_link") |
                                        (df["Model Name"] == "rover_argo_NZero::rear_left_wheel_link")].index)        
    equipment_lat = df_filtered["Latitude"]
    equipment_lon = df_filtered["Longitude"]

    points_to_use_quadtree = np.array([equipment_lon, equipment_lat]).T

    return points_to_use_quadtree

def LoadBoundaryPoints(points_to_use_quadtree):
    """
    Load boundary points for quadtree.

    Args:
        points_to_use_quadtree (np.array): Points to use in the quadtree.
    
    Returns:
        tuple: Boundary and points to use in the quadtree
    """

    # Min and max latitude and longitude
    min_lat = points_to_use_quadtree[:, 1].min()
    max_lat = points_to_use_quadtree[:, 1].max()
    min_lon = points_to_use_quadtree[:, 0].min()
    max_lon = points_to_use_quadtree[:, 0].max()

    # Create quadtree
    boundary = (min_lon, min_lat, max_lon, max_lat) # Area boundary

    return boundary

def PlotQuadtree(ax, bound, points, color='b', edge=True, capacity=4):
    """
    Plot quadtree.
    
    Args:
        boundary (tuple): Boundary
        points_to_use_quadtree (np.array): Points to use in the quadtree.
    
    Returns:
        None"""
        
    # Create quadtree
    quad = Quadtree(bound, capacity=capacity)

    for point in points_to_use_quadtree:
        quad.Insert(point)

    if edge:
        quad.Plot(ax)

    ax.scatter(points[:, 0], points[:, 1], color=color,s=6)
    ax.set_aspect('equal')

####################################################################################################
# MAIN FUNCTION
####################################################################################################

if __name__ == "__main__":

    # Load equipment points and Voronoi vertices
    df = LoadFileToDataframe()
    voronoi_diagram = voronoi.VoronoiDiagram()
    vertices = voronoi_diagram.PlotVoronoiDiagram(return_ver=True)
    points_to_use_quadtree = FilterDataframe(df)
    
    fig, ax = plt.subplots()

    # Plot quadtree
    boundary_eq = LoadBoundaryPoints(points_to_use_quadtree)
    PlotQuadtree(ax, boundary_eq, points_to_use_quadtree,edge=False)

    boundary_ver = LoadBoundaryPoints(vertices)
    PlotQuadtree(ax, boundary_ver, vertices, color='orange', capacity=10)

    plt.show()