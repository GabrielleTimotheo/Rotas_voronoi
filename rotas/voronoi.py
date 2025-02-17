import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import pandas as pd
from scipy.spatial import ConvexHull

class VoronoiDiagram:
    def __init__(self, file_path = "models.xlsx"):
        """
        Initialize Voronoi Diagram.

        Args:
            file_path (str): File path
        """

        # Load file
        self.file_path = file_path
        self.df = self.LoadFileToDataframe()

        # Filter dataframe
        self.FilterDataframe()

        # Adjust initial point
        self.AdjustInitialPoint()

        # Adjust information to plot
        self.points_to_plot = np.array([self.equipment_lon, self.equipment_lat]).T
        self.points_to_plot = np.concatenate((self.points_to_plot, self.bounding_box_points_to_plot), axis=0)

        # Add extra points to the diagram
        # self.points_to_plot = self.RightHandRuleOrder(self.points_to_plot)

        # Apply Quadtree
        # self.Quadtree()

        # Create voronoi diagram to plot
        self.vor = Voronoi(self.points_to_plot)

        # Real points and real voronoi diagram
        self.real_points = np.array([self.equipment_lat, self.equipment_lon]).T
        self.real_points = np.concatenate((self.real_points, self.real_bounding_box_points), axis=0)

        # Add extra points to the real diagram
        # self.real_points = self.AddExtraPoints(self.real_points)

        self.real_vor = Voronoi(self.real_points)

        # Collect sites, only central points inside the cells
        self.voronoi_sites = pd.DataFrame(self.real_vor.points, columns=["Latitude", "Longitude"])

    def LoadFileToDataframe(self):
        """
        Load file to pandas dataframe.

        Args:
            file_path (str): File path
        Returns:
            pd.DataFrame: Dataframe
        """
        try:
            df = pd.read_excel(self.file_path)
            return df
        except Exception as e:
            print(f"Could not load the file: {e}")
            return None

    def FilterDataframe(self):
        """
        Filter the dataframe to get only the necessary information and find the robot initial point.
        """
        # Remove extra robot coordinates
        self.df_filtered = self.df.drop(self.df[(self.df["Model Name"] == "rover_argo_NZero::base_link") | 
                                                (self.df["Model Name"] == "rover_argo_NZero::front_left_wheel_link") | 
                                                (self.df["Model Name"] == "rover_argo_NZero::front_right_wheel_link") |
                                                (self.df["Model Name"] == "rover_argo_NZero::rear_right_wheel_link") |
                                                (self.df["Model Name"] == "rover_argo_NZero::rear_left_wheel_link")].index)        
        self.model_name = self.df_filtered["Model Name"]
        self.equipment_lat = self.df_filtered["Latitude"]
        self.equipment_lon = self.df_filtered["Longitude"]

        # Initial point where the robot is located
        self.initial_point = self.df.loc[self.df["Model Name"] == "rover_argo_NZero::imu_link"] # Find the related row 
        self.initial_point_lat = self.initial_point["Latitude"].values[0]
        self.initial_point_lon = self.initial_point["Longitude"].values[0]
    
    def AdjustInitialPoint(self):
        """
        Create bounding box for the initial point.
        """

        # Define the bounding box of the initial point
        self.initial_point_lat_upper_left = self.initial_point_lat + 0.00001
        self.initial_point_lon_upper_left = self.initial_point_lon - 0.00001
        self.initial_point_lat_upper_right = self.initial_point_lat + 0.00001
        self.initial_point_lon_upper_right = self.initial_point_lon + 0.00001
        self.initial_point_lat_lower_right = self.initial_point_lat - 0.00001
        self.initial_point_lon_lower_right = self.initial_point_lon + 0.00001
        self.initial_point_lat_lower_left = self.initial_point_lat - 0.00001
        self.initial_point_lon_lower_left = self.initial_point_lon - 0.00001
        
        # Create an array with the bounding box points
        self.bounding_box_points_to_plot = np.array([
            [self.initial_point_lon_upper_left, self.initial_point_lat_upper_left],
            [self.initial_point_lon_upper_right, self.initial_point_lat_upper_right],
            [self.initial_point_lon_lower_right, self.initial_point_lat_lower_right],
            [self.initial_point_lon_lower_left, self.initial_point_lat_lower_left]
        ])

        self.real_bounding_box_points = np.array([
            [self.initial_point_lat_upper_left, self.initial_point_lon_upper_left],
            [self.initial_point_lat_upper_right, self.initial_point_lon_upper_right],
            [self.initial_point_lat_lower_right, self.initial_point_lon_lower_right],
            [self.initial_point_lat_lower_left, self.initial_point_lon_lower_left]
        ])

        # Initial point
        self.initial_point = np.array([self.initial_point_lat, self.initial_point_lon])
    
    # def Quadtree(self):

    #     # Filter data so as not to add the initial point
    #     df_quadtree = self.df.drop(self.df[(self.df["Model Name"] == "rover_argo_NZero::imu_link")].index)

    #     # Points
    #     self.points_to_use_quadtree = np.array([df_quadtree["Longitude"], df_quadtree["Latitude"]]).T

    #     # Min and max latitude and longitude
    #     min_lat = self.points_to_use_quadtree[:, 1].min()
    #     max_lat = self.points_to_use_quadtree[:, 1].max()
    #     min_lon = self.points_to_use_quadtree[:, 0].min()
    #     max_lon = self.points_to_use_quadtree[:, 0].max()

    #     boundary = (min_lon, min_lat, max_lon, max_lat) 
    #     capacity = 5

    #     # Apply Quadtree to the points 
    #     self.quadtree = quadtree.Quadtree(boundary, capacity)

    #     for point in self.points_to_use_quadtree:
    #         self.quadtree.insert(point)

    #     # Plota a quadtree
    #     fig, ax = plt.subplots()
    #     self.quadtree.plot(ax)
    #     ax.scatter(self.points_to_use_quadtree[:, 0], self.points_to_use_quadtree[:, 1], color='red')
    #     ax.set_aspect('equal')
    #     plt.show()

    def PlotVoronoiDiagram(self):
        """
        Plot voronoi diagram.
        """
        fig, ax = plt.subplots()
        voronoi_plot_2d(self.vor, ax=ax)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Voronoi Diagram")
        # plt.show()

    def GenerateExtraPoints(self, points, expansion_factor=0.01):
        """
        Generate extra points around the convex hull of the given points.

        Args:
            points (np.array): Array of points.
            expansion_factor (float): Factor to expand the convex hull.
        Returns:
            np.array: Extra points.
        """
        # Calculate the convex hull of the points
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        # Calculate the centroid of the hull points
        centroid = np.mean(hull_points, axis=0)

        # Generate extra points by expanding the hull points away from the centroid
        extra_points = centroid + (hull_points - centroid) * (1 + expansion_factor)

        return extra_points


    # def AddExtraPoints(self, points, spread=0.0001, num_extra=100):
    #     """
    #     Add extra points to the diagram.
        
    #     Args:
    #         points (np.array): Array of points.
    #         spread (float): Spread of the extra points.
    #         num_extra (int): Number of extra points.
    #     Returns:
    #         np.array: Array of points with extra points.
    #     """
        
    #     hull = ConvexHull(points)
    #     hull_points = points[hull.vertices]
        
    #     extra_points = []
    #     for point in hull_points:
    #         for _ in range(num_extra // len(hull_points)): # Distribute the extra points evenly among the hull points
    #             offset = np.random.uniform(-spread, spread, size=2) # Deslocamento aleatóri na horizontal e vertical
    #             extra_points.append(point + offset)
        
    #     extra_points = np.array(extra_points)
        
    #     return np.vstack([points, extra_points])