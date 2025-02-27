import numpy as np
import matplotlib.pyplot as plt

class BoundaryPoints:
    def __init__(self):
        """
        Initialize BoundaryPoints.
        """
        self.selected_points = self.ReadSelectedPoints()
        self.new_points, self.centro = self.AddExtraPoints()
        self.interpolated_points = self.GenerateIntermediatePoints(num=7)

    def ReadSelectedPoints(self):
        """
        Read selected points from a text file.

        Args:
            file_path (str): Path to the text file.

        Returns:
            np.ndarray: Array of selected points.
        """
        try:
            points = np.loadtxt("edge_stitches.txt", delimiter=",", dtype=float)
            points = np.array(points)
            return points
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

    def AddExtraPoints(self):
        """
        Add extra points to the selected points.

        Args:
            points (np.ndarray): Array of selected points.

        Returns:
            np.ndarray: Array of selected points with extra points.
        """
        # Fator de afastamento
        k = 0.1  

        # Calcula o centroide
        centro = np.mean(self.selected_points, axis=0)

        # Calcula os novos pontos afastados
        new_points = []
        for p in self.selected_points:
            direction = p - centro
            new_point = p + k * direction
            new_points.append(new_point)

        new_points = np.array(new_points)
        return new_points, centro

    def GenerateIntermediatePoints(self, num=5):
        """
        Generate intermediate points between the selected points.

        Args:
            points (np.ndarray): Array of selected points.
            num (int): Number of intermediate points to generate.
        
        Returns:
            np.ndarray: Array of intermediate points.
        """
        interpolated = []
        for i in range(len(self.new_points)):
            p1 = self.new_points[i]
            p2 = self.new_points[(i + 1) % len(self.new_points)]  # Conecta o último ao primeiro (fechando o ciclo)
            
            for t in np.linspace(0, 1, num, endpoint=False)[1:]:  # Ignora t=0 (o próprio ponto p1)
                interpolated.append((1 - t) * p1 + t * p2)
        
        return np.array(interpolated)

    def PlotPoints(self):
        """
        Plot selected points and new points.

        Args:
            selected_points (np.ndarray): Array of selected points.
            new_points (np.ndarray): Array of new points.
            centro (np.ndarray): Centroid of the selected points.

        Returns:
            None
        """
        # Plotando os pontos antes e depois
        plt.scatter(self.selected_points[:,0], self.selected_points[:,1], color='blue', label='Original')
        plt.scatter(self.interpolated_points[:,0], self.interpolated_points[:,1], color='green', s=10, label='Interpolado')
        plt.scatter(self.centro[0], self.centro[1], color='black', marker='x', label='Centroide')
        plt.legend()
        plt.grid()
        plt.show()