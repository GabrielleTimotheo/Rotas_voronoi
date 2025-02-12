import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import heapq
import voronoi

class AStarGraph:
    def __init__(self):
        """
        Initialize A* Graph.
        """

        # Load Voronoi Diagram
        self.voronoi_diagram = voronoi.VoronoiDiagram()
        self.vor_ver_index = self.voronoi_diagram.real_vor.ridge_vertices # vertices index
        self.vor_vertices = self.voronoi_diagram.real_vor.vertices # vertices coordinates
        self.voronoi_sites = self.voronoi_diagram.voronoi_sites
        self.voronoi_sites = self.voronoi_sites.to_numpy()

        # self.start_usuario = self.voronoi_diagram.initial_point # test point
        # self.goal_usuario = self.voronoi_sites[5] # test point, 10 is an example

        self.selected_points = [] # Store the selected points
        self.equipment_lon = self.voronoi_diagram.equipment_lon.to_numpy()
        self.equipment_lat = self.voronoi_diagram.equipment_lat.to_numpy()

        self.PlotEquipamentCoordinates()

        self.start_usuario = self.selected_points[0]
        self.goal_usuario = self.selected_points[1]

        # self.goal_usuario = [-3.1236522876070354,-41.764444401709717]

        # Find the closest vertex to the start point and goal point
        self.start = self.FindClosestVertex(self.start_usuario)
        self.goal = self.FindClosestVertex(self.goal_usuario)

        # Create a graph for path planning
        self.graph = self.BuildGraph()

    def on_click(self, event):
        """
        Capture the click and store the coordinate of the clicked point.

        Args:
            event (event): Event
        """
        if event.xdata is not None and event.ydata is not None:
            clicked_point = (event.xdata, event.ydata)
            self.selected_points.append(clicked_point)
            print(f"Coordenada selecionada: {clicked_point}")

    def on_click(self, event):
        """
        Capture the click and store the coordinate of the clicked point.

        Args:
            event (event): Event
        """
        if event.xdata is not None and event.ydata is not None:
            clicked_point = (event.xdata, event.ydata)
            
            # Calculate the Euclidean distance between the clicked point and all equipment points
            distances = np.sqrt((self.equipment_lon - clicked_point[0])**2 + (self.equipment_lat - clicked_point[1])**2)
            
            # Find the index of the closest point
            closest_index = np.argmin(distances)
            
            # Get the closest point coordinates
            closest_point = (self.equipment_lat[closest_index], self.equipment_lon[closest_index])
            
            self.selected_points.append(closest_point)
            print(f"Coordenada selecionada: {closest_point}")

    def PlotEquipamentCoordinates(self):
        """
        Plot equipament coordinates and collect points to the mission.
        """
        fig, ax = plt.subplots()
        ax.scatter(self.equipment_lon, self.equipment_lat, picker=True)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Select the initial point and then, equipment to send the mission")

        # Conecte the event with a click
        fig.canvas.mpl_connect('button_press_event', self.on_click)

        plt.show()

    def FindClosestVertex(self, usuario_point):
        """
        Find the closest vertex to the start point.

        Args:
            usuario_point (tuple): Start point
        
        Returns:
            tuple: Closest vertex
        """
        min_distance = float('inf')
        closest_vertex = None

        for vertex in self.vor_vertices:
            distance = self.EuclideanDistance(usuario_point, vertex)
            if distance < min_distance:
                min_distance = distance
                closest_vertex = vertex

        return closest_vertex

    def BuildGraph(self):
        """
        Build graph for path planning based on Voronoi diagram.

        Returns:
            nx.Graph: Graph
        """

        # Create a graph for path planning
        graph = nx.Graph()

        # Add Voronoi diagram edges to the graph and weight them by Euclidean distance
        for start, end in self.vor_ver_index:
            if start >= 0 and end >= 0:  # Filter invalid vertices
                p1, p2 = self.vor_vertices[start], self.vor_vertices[end]
                distance = np.linalg.norm(p1 - p2)  # Euclidean distance
                graph.add_edge(tuple(p1), tuple(p2), weight=distance)
        
        return graph

    def EuclideanDistance(self, a, b):
        """
        Heuristic function: Euclidean distance (h(n)).

        Args:
            a (tuple): Point A
            b (tuple): Point B
        Returns:
            float: Euclidean distance between points A and B
        """
        return np.linalg.norm(np.array(a) - np.array(b))

    def AStar(self):
        """
        A* search algorithm.

        Returns:
            list: Path found by A* algorithm
        """

        open_set = [] # store edges to be explored
        heapq.heappush(open_set, (0, tuple(self.start))) # (f(n), node)
        
        came_from = {}
        g_score = {tuple(self.start): 0}
        f_score = {tuple(self.start): self.EuclideanDistance(self.start, self.goal)}
        
        while open_set:
            current_f, current_node = heapq.heappop(open_set)
            
            # If we reached the goal, reconstruct the path
            if np.array_equal(np.array(current_node), self.goal):
                path = []
                while current_node in came_from:
                    path.append(current_node)
                    current_node = came_from[current_node]
                path.append(tuple(self.start))
                path.reverse()

                # Add start and goal points to the path
                # path.insert(0, tuple(self.start_usuario))
                # path.append(tuple(self.goal_usuario))

                return path, self.voronoi_sites
            
            # Verify neighbors
            for neighbor in self.graph.neighbors(current_node):
                tentative_g_score = g_score[current_node] + self.graph[current_node][neighbor]["weight"]
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.EuclideanDistance(neighbor, self.goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found

    def PlotPath(self, path):
        """
        Plot path found by A*.
        """
        
        if path:
            path = np.array(path)
            plt.plot(path[:, 1],path[:, 0], color='yellow', linewidth=2, label="Caminho (A*)")
            plt.scatter(self.start_usuario[1], self.start_usuario[0], color='green', s=100, label="Start")
            plt.scatter(self.goal_usuario[1], self.goal_usuario[0], color='purple', s=100, label="Goal")
            plt.scatter(self.voronoi_sites[:, 1], self.voronoi_sites[:, 0], color='blue')

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Diagrama de Voronoi com Caminho A*")
        plt.legend()
        # plt.show()
