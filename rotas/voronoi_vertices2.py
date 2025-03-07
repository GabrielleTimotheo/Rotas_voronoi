import numpy as np
import matplotlib.pyplot as plt
import voronoi
import networkx as nx
from sklearn.cluster import DBSCAN

####################################################################################################
# SMOOTHING VORONOI VERTICES
####################################################################################################

class VoronoiVertices:
    
    def __init__(self):
        
        # Load Voronoi Diagram
        self.voronoi_diagram = voronoi.VoronoiDiagram()
        self.vor_ver_index = self.voronoi_diagram.vor.ridge_vertices # vertices index
        self.vor_vertices = self.voronoi_diagram.vor.vertices # vertices coordinates

        self.vertices = self.voronoi_diagram.PlotVoronoiDiagram(return_ver=True)#[:200]

        # Apply DBSCAN algorithm
        self.Clustering()

        # Create a graph for path planning
        self.graph = self.BuildGraph()

        # Simplify paths in the graph
        self.SimplifyPaths()
        self.PlotGraph()

    def Clustering(self):
        """
        Apply Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm.
        """

        db = DBSCAN(eps=0.0001, min_samples=5).fit(self.vertices)
        labels = db.labels_ # Labels from each point (identified cluster)        
        n_clusters = len(np.unique(labels[labels != -1])) # Number of clusters in labels, ignoring noise if present

        # Get the new vertices after clustering (excluding noise points)
        self.filtered_vertices = self.vertices[labels != -1]

        # Plotando os clusters
        plt.figure(figsize=(8, 6))
        unique_labels = set(labels)
        colors = plt.cm.get_cmap("tab20", len(unique_labels))  # Gera cores diferentes

        for label in unique_labels:
            if label == -1:
                color = "k"  # Cor preta para ruído
                marker = "x"
            else:
                color = colors(label)
                marker = "o"

            # Seleciona os pontos pertencentes ao cluster
            cluster_points = self.vertices[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                        c=[color], label=f"Cluster {label}" if label != -1 else "Ruído",
                        marker=marker, edgecolors="k", alpha=0.6)

        plt.title(f"DBSCAN Clustering ({n_clusters} clusters detectados)")
        plt.xlabel("Coordenada X")
        plt.ylabel("Coordenada Y")
        plt.legend()
        plt.grid()
        plt.show()


        # Aplicar Ramer-Douglas-Peucker em cada cluster
        self.simplified_clusters = []
        for label in unique_labels:
            if label != -1:  # Ignora ruído
                cluster_points = self.vertices[labels == label]
                simplified_points = self.RamerDouglasPeucker(cluster_points, epsilon=0.001)
                self.simplified_clusters.append(simplified_points)

    def BuildGraph(self):
        """
        Build a graph for path planning based on Voronoi diagram.
        Builds the graph only between the new vertices after clustering.
        """
        # Create a graph for path planning
        graph = nx.Graph()

        # Create a mapping from original Voronoi vertices to filtered vertices
        valid_vertex_indices = [
            i for i, point in enumerate(self.vor_vertices) 
            if any(np.array_equal(point, filtered_point) for filtered_point in self.filtered_vertices)
        ]
        
        # Add edges between filtered vertices
        for start, end in self.vor_ver_index:
            # Ensure both vertices are in the filtered vertices list
            if start >= 0 and end >= 0 and start in valid_vertex_indices and end in valid_vertex_indices:
                p1, p2 = self.vor_vertices[start], self.vor_vertices[end]
                distance = np.linalg.norm(p1 - p2)  # Euclidean distance
                graph.add_edge(tuple(p1), tuple(p2), weight=distance)
        
        return graph
    
    def SimplifyPaths(self):
        """
        Simplify the paths in each cluster using Ramer-Douglas-Peucker.
        """
        # Simplifica as rotas para cada cluster
        self.simplified_paths = []
        for cluster in self.simplified_clusters:
            simplified_path = self.RamerDouglasPeucker(cluster, epsilon=0.001)
            self.simplified_paths.append(simplified_path)

    def PerpendicularDistance(self, point, line_start, line_end):
        """Calcula a distância perpendicular de um ponto a uma linha definida por dois pontos."""
        if np.array_equal(line_start, line_end):
            return np.linalg.norm(point - line_start)
        
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.dot(line_vec, line_vec)
        projection = np.dot(point_vec, line_vec) / line_len
        projection = np.clip(projection, 0, 1)
        closest_point = line_start + projection * line_vec
        return np.linalg.norm(point - closest_point)

    def RamerDouglasPeucker(self, points, epsilon=0.001):
        """Reduz a quantidade de pontos usando o algoritmo Ramer-Douglas-Peucker."""
        if len(points) < 3:
            return points
        
        points = np.array(points)
        start, end = points[0], points[-1]
        
        # Encontra o ponto mais distante da linha reta
        distances = np.array([self.PerpendicularDistance(p, start, end) for p in points])
        max_index = np.argmax(distances)
        max_distance = distances[max_index]
        
        if max_distance > epsilon:
            # Recursivamente simplifica os dois segmentos
            left_simplified = self.RamerDouglasPeucker(points[:max_index+1], epsilon)
            right_simplified = self.RamerDouglasPeucker(points[max_index:], epsilon)
            return np.vstack((left_simplified[:-1], right_simplified))
        else:
            return np.array([start, end])
        
    def PlotGraph(self):
        """
        Plot the graph created for path planning using NetworkX and Matplotlib.
        """
        plt.figure(figsize=(8, 6))

        # Define positions of the nodes based on their coordinates
        positions = {tuple(p): p for p in self.filtered_vertices}

        # Draw nodes and edges
        nx.draw(self.graph, pos=positions, with_labels=False, node_size=50, node_color='blue', edge_color='gray', alpha=0.7, font_size=8)

        plt.title("Graph for Path Planning")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid()
        plt.show()

# Exemplo de uso:
if __name__ == "__main__":

    # Create VoronoiVertices object
    voronoi_vertices = VoronoiVertices()


    # path = np.array([(0, 0), (1, 0.1), (2, -0.1), (3, 5), (4, 6), (5, 7), (6, 8)])
    # epsilon = 2.0
    # simplified_path = ramer_douglas_peucker(path, epsilon)
    
    # # Plot original path
    # plt.plot(path[:, 0], path[:, 1], 'b-', label='Original Path')
    
    # # Plot simplified path
    # plt.plot(simplified_path[:, 0], simplified_path[:, 1], 'r-', label='Simplified Path')
    
    # plt.legend()
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Ramer-Douglas-Peucker Algorithm')
    # plt.show()