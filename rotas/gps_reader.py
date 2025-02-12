import rospy
import matplotlib
import matplotlib.pyplot as plt
from sensor_msgs.msg import NavSatFix
import numpy as np
import voronoi

# Garantir que o Matplotlib está rodando no backend correto
matplotlib.use("TkAgg")

class RobotGPSPlotter:
    def __init__(self):
        """
        Initialize the node and the plot.
        """
        rospy.init_node('robot_gps_plotter', anonymous=True)

        self.latitude = []
        self.longitude = []

        self.vor = voronoi.VoronoiDiagram()
        self.voronoi_sites = self.vor.voronoi_sites
        self.voronoi_sites = self.voronoi_sites.to_numpy()

        self.path = np.loadtxt("path.txt", delimiter=",", dtype=float)
        self.path = np.array(self.path)
        print(self.path)

        rospy.Subscriber('/mavros/global_position/global', NavSatFix, self.gps_callback)

    def gps_callback(self, data):
        """
        Receive the GPS data and store it for later plotting.
        
        Args:
            data (NavSatFix): GPS data
        """
        print("Callback acionado!")
        if data.status.status >= 0:  # Verifica se o GPS está válido
            print(f"Recebendo GPS: {data.latitude}, {data.longitude}")  # Debug
            self.latitude.append(data.latitude)
            self.longitude.append(data.longitude)

    def plot_gps_data(self):
        """
        Plot the collected GPS data.
        """

        # Plotar os dados coletados
        plt.plot(self.path[:, 1],self.path[:, 0], color='yellow', linewidth=2, label="Caminho (A*)")
        plt.scatter(self.voronoi_sites[:, 1], self.voronoi_sites[:, 0], color='blue')
        plt.scatter(self.longitude, self.latitude, color='red')
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("GPS Data Collected")
        plt.show()

    def run(self):
        """ 
        Keep the node running and collect data.
        """
        rospy.spin()  # Aguarda a coleta de dados, para então exibir o gráfico ao finalizar

if __name__ == '__main__':
    try:
        plotter = RobotGPSPlotter()
        plotter.run()

        # Quando a execução do rospy for encerrada, plota os dados
        plotter.plot_gps_data()

    except rospy.ROSInterruptException:
        pass
