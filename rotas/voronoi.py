import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import pandas as pd

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

        # Filter data
        # self.df_filtered = self.df.drop(index=[429, 430, 431, 432, 433])  
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
        self.initial_point = np.array([self.initial_point_lat, self.initial_point_lon])

        # Adjust information to plot
        self.points_to_plot = np.array([self.equipment_lon, self.equipment_lat]).T

        # Create voronoi diagram to plot
        self.vor = Voronoi(self.points_to_plot)

        # Real points and real voronoi diagram
        self.real_points = np.array([self.equipment_lat, self.equipment_lon]).T
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
