import numpy as np
import matplotlib.pyplot as plt 
import voronoi

class Quadtree:
    def __init__(self, boundary, capacity=4):
        """
        Initialize Quadtree.

        Args:
            boundary (tuple): Area boundary.
            capacity (int): Maximum number of points before subdividing.
        """

        self.boundary = boundary # Area boundary
        self.capacity = capacity  # Maximum number of points before subdividing
        self.points = []  # List of points in the quadtree
        self.divided = False  # Indicative if the quadtree is divided
        self.quadrants = []  # Quadrants

    def insert(self, point):
        """
        Insert a point into the quadtree.
        
        Args:
            point (tuple): Point to be inserted.
        
        Returns:
            bool: True if the point was inserted, False otherwise.
        """
        if not self._in_boundary(point):
            return False
        
        if len(self.points) < self.capacity:
            self.points.append(point)
            return True

        if not self.divided:
            self._subdivide()

        # for quadrant in self.quadrants:
        #     if quadrant.insert(point):
        #         return True

        # Redistribute points to quadrants
        for p in self.points:
            for quadrant in self.quadrants:
                if quadrant._in_boundary(p):
                    quadrant.insert(p)
        self.points = []  # Clear points in this node after redistribution

        for quadrant in self.quadrants:
            if quadrant.insert(point):
                return True
        
        return False

    def _in_boundary(self, point):
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
        # return xmin < x < xmax and ymin < y < ymax

    def _subdivide(self):
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
    
    def plot(self, ax):
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
                quadrant.plot(ax) 
    
    def get_vertices(self):
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
    
# Clean selected points file
open("selected_points.txt", "w").close()

# Load Voronoi Diagram and get data
voronoi_diagram = voronoi.VoronoiDiagram()
points_to_use_quadtree = voronoi_diagram.points_to_plot

# Min and max latitude and longitude
min_lat = points_to_use_quadtree[:, 1].min()
max_lat = points_to_use_quadtree[:, 1].max()
min_lon = points_to_use_quadtree[:, 0].min()
max_lon = points_to_use_quadtree[:, 0].max()

# Apply Quadtree to the points 
quadtree = Quadtree((min_lon, min_lat, max_lon, max_lat))

for point in points_to_use_quadtree:
    quadtree.insert(point)

# Plot
fig, ax = plt.subplots()
quadtree.plot(ax)
ax.scatter(points_to_use_quadtree[:, 0], points_to_use_quadtree[:, 1], color='red')
ax.set_aspect('equal')

# Interactive point selection
selected_points = []
vertices = np.array(quadtree.get_vertices())

def on_click(event):
    if event.xdata is None or event.ydata is None:
        return
    
    click_point = np.array([event.xdata, event.ydata])
    distances = np.linalg.norm(vertices - click_point, axis=1)
    nearest_vertex = vertices[np.argmin(distances)]
    
    selected_points.append(nearest_vertex)
    ax.scatter(nearest_vertex[0], nearest_vertex[1], color='blue', marker='x')
    plt.draw()
    
    with open("selected_points.txt", "a") as f:
        f.write(f"{nearest_vertex[0]}, {nearest_vertex[1]}\n")

fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()
