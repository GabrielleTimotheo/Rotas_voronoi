import voronoi
from scipy.spatial import distance
from shapely.geometry import Point, LineString
import numpy as np
import matplotlib.pyplot as plt

class VoronoiVertices:
    def __init__(self, vertices, ridge_vertices, points):

        self.vertices = vertices # Voronoi vertices
        self.ridge_vertices = ridge_vertices # Neighbour vertices
        self.points = points # Equipment points

        # Neighbour set
        # self.neighbor_set = self.BuildNeighborSet()

        self.proximity = 0.0001  # Proximity threshold

        # Lines
        self.lines = []

    # def BuildNeighborSet(self):
    #     """
    #     Build a set with the neighbours of each vertex.
        
    #     Returns:
    #         set: Set with the neighbours of each vertex.
    #     """
        
    #     # Create a set to store the neighbors without repetitions
    #     neighbors = set()

    #     # Iterate over the ridge vertices
    #     for ridge in self.ridge_vertices:

    #         if -1 in ridge:  # Ignore the infinite vertices
    #             continue

    #         neighbors.add(frozenset(ridge))

    #     return neighbors
    
    # def CheckIfNeighbour(self, v1, v2):
    #     """
    #     Check if two vertices are neighbours.

    #     Args:
    #         v1 (tuple): First vertex.
    #         v2 (tuple): Second vertex.
        
    #     Returns:
    #         bool: True if the vertices are neighbours, False otherwise.
    #     """

    #     if frozenset([v1, v2]) in self.neighbor_set:
    #         return True
    #     else:
    #         return False
    
    def GetLineEquation(self, p1, p2):
        """
        Get the equation of the line between two points.
        
        Args:
            p1 (tuple): First point.
            p2 (tuple): Second point.
        
        Returns:
            tuple: Slope and intercept of the line.
        """

        # Points
        x1, y1 = p1
        x2, y2 = p2

        if x1 == x2:
            return None, x1

        # Slope and intercept
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1 
        return m, b

    def CrossesEquipment(self, m, b, p1, p2, safety_radius=0.00008):
        """
        Check if a line crosses the equipment.
        
        Args:
            m (float): Slope of the line.
            b (float): Intercept of the line.
        
        Returns:
            bool: True if the line crosses the equipment, False otherwise.
        """

        if m is None:  # Linha vertical
            for point in self.points:
                x, y = point
                if abs(x - b) <= safety_radius:  
                    return True
            return False
        
        elif m == 0:  # Linha horizontal
            for point in self.points:
                x, y = point
                if abs(y - b) <= safety_radius:
                    return True
            return False

        # Para linhas não verticais, a equação y = mx + b é válida
        for point in self.points:
            x, y = point
            distance_to_line = self.PointToLineDistance(point, [p1, p2])  # Distância do ponto à linha
            if distance_to_line <= safety_radius:  # Verifica se a distância é menor que o raio da bolha
                return True
        return False
    
    def CheckIfThereIsVerticeNearby(self, p1, p2):
        """
        Check if there is a vertice nearby.
        
        Args:
            p1 (tuple): First point.
            p2 (tuple): Second point.
        
        Returns:
            bool: True if there is a vertice nearby, False otherwise.
        """
        i = 0

        # Iterate through existing lines
        for ver in self.vertices:

            if np.array_equal(ver, p1) and np.array_equal(ver, p2):
                continue

            distance = self.PointToLineDistance(ver, (p1, p2))

            if distance < 0.00006:

                # Check if the point is between p1 and p2 on the segment
                # Use vector cross product to check if the point is on the segment
                # Vector p1 -> ver and p1 -> p2 should have the same direction
                vec1 = np.array(ver) - np.array(p1)
                vec2 = np.array(p2) - np.array(p1)
                
                # Check if the ver lies within the bounds of p1 and p2 in both x and y
                # Produto escalar
                if np.dot(vec1, vec2) >= 0 and np.linalg.norm(vec1) <= np.linalg.norm(vec2):

                    i = i + 1

                    if i == 2:
                        return True
        
        # If no vertice is nearby, return False
        return False
        
    def EuclideanDistance(self, p1, p2):
        """
        Calculate the euclidean distance between two points.
        
        Args:
            p1 (tuple): First point.
            p2 (tuple): Second point.
        
        Returns:
            float: Euclidean distance between the points.
        """

        return distance.euclidean(p1, p2)

    def PointToLineDistance(self, point, line):
        """
        Calculate the distance between a point and a line.
        
        Args:
            point (tuple): Point.
            line (tuple): Line.
        
        Returns:
            float: Distance between the point and the line.
        """

        line_geom = LineString([line[0], line[1]])
        return line_geom.distance(Point(point))
    
    def CheckLine(self, p1, p2):
        """
        Check if a new line can replace an existing one or if it is redundant.
        
        Args:
            p1 (tuple): First point.
            p2 (tuple): Second point.
        
        Returns:
            bool: True if the new line should be added, False otherwise.
        """

        new_line_distance = self.EuclideanDistance(p1, p2)
        new_line_geom = LineString([p1, p2])

        for i, existing_line in enumerate(self.lines):

            existing_line_geom = LineString(existing_line)

            # Change line to delete the inicial and final points
            buffer_distance = 0.0001
            new_line_buffered = new_line_geom.buffer(-buffer_distance)
            existing_line_buffered = existing_line_geom.buffer(-buffer_distance)

            # Calculate the distance between the lines
            if new_line_buffered.intersects(existing_line_buffered):
                existing_line_distance = self.EuclideanDistance(existing_line[0], existing_line[1])

                # Keep the longer line
                if new_line_distance > existing_line_distance:
                    self.lines[i] = [p1, p2]
                return False

            # Calculate midpoint and distance to the new line
            midpoint_existing = np.mean([existing_line[0], existing_line[1]], axis=0)
            distance_to_new_line = self.PointToLineDistance(midpoint_existing, (p1, p2))

            # Calculate angle difference (dot product of direction vectors)
            vec1 = np.array(p2) - np.array(p1)
            vec2 = np.array(existing_line[1]) - np.array(existing_line[0])
            angle_diff = np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0))

            angle_threshold = np.radians(60) 

            # If they are too close and have similar angles, discard the new line
            if distance_to_new_line < self.proximity and abs(angle_diff) < angle_threshold:
                existing_line_distance = self.EuclideanDistance(existing_line[0], existing_line[1])

                # Keep the longer line
                if new_line_distance > existing_line_distance:
                    self.lines[i] = [p1, p2]

                return False

        # If no similar line was found, add the new one
        return True


    def CreateLines(self):
        """
        Create lines between the voronoi vertices.
        
        Returns:
            list: List with the lines between the voronoi vertices.
        """

        num_vertices = len(self.vertices)
        
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):  

                # Check if the vertices are neighbours
                # if self.CheckIfNeighbour(i, j): 
                #     continue

                p1 = self.vertices[i]
                p2 = self.vertices[j]

                if not self.CheckIfThereIsVerticeNearby(p1, p2):
                    continue

                if self.CrossesEquipment(*self.GetLineEquation(p1, p2), p1, p2):
                    continue
                
                if len(self.lines) >= 2:
                    self.CheckLine(p1, p2)
                else:
                    self.lines.append([p1, p2])

                # Add the line
                if self.CheckLine(p1, p2):
                    self.lines.append([p1, p2])  

                print("self.lines:", len(self.lines))
    
    def PlotLines(self):
        """
        Plot the lines between the voronoi vertices.
        
        Returns:
            None
        """

        # Plot the lines
        for line in self.lines:
            plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], color="red", alpha=0.5)

        # Plot equipment points
        plt.scatter(self.points[:, 0], self.points[:, 1], color="blue", label="Equipment points")

        plt.show()

if __name__ == "__main__":

    # Create voronoi diagram
    vor = voronoi.VoronoiDiagram()

    # Voronoi vertices
    vertices = vor.vor.vertices
    vertices = vor.PlotVoronoiDiagram(return_ver=True)[:400]

    # Neighbour vertices
    ridge_vertices = vor.vor.ridge_vertices

    # Equipment points
    points = vor.points_to_plot

    # Create VoronoiVertices object
    voronoi_vertices = VoronoiVertices(vertices, ridge_vertices, points)
    voronoi_vertices.CreateLines()
    voronoi_vertices.PlotLines()