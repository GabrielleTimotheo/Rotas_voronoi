import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import pandas as pd
from scipy.spatial import ConvexHull
import treat_extra_points

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

        # Add points to the bounding box
        self.treat_extra_points = treat_extra_points.BoundaryPoints()
        self.points_to_plot_extra = np.concatenate((self.points_to_plot, self.treat_extra_points.new_points, self.treat_extra_points.interpolated_points), axis=0)

        # Create voronoi diagram to plot
        self.vor = Voronoi(self.points_to_plot_extra)

        # Extra points
        self.lon_point1 = self.treat_extra_points.new_points[:, 1]
        self.lat_point1 = self.treat_extra_points.new_points[:, 0]
        self.extra_points1 = np.array([self.lon_point1, self.lat_point1]).T

        self.lon_point2 = self.treat_extra_points.interpolated_points[:, 1]
        self.lat_point2 = self.treat_extra_points.interpolated_points[:, 0]
        self.extra_points2 = np.array([self.lon_point2, self.lat_point2]).T

        # Real points and real voronoi diagram
        self.real_points = np.array([self.equipment_lat, self.equipment_lon]).T
        self.real_points = np.concatenate((self.real_points, self.real_bounding_box_points), axis=0)
        self.real_points = np.concatenate((self.real_points, self.extra_points1, self.extra_points2), axis=0)

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