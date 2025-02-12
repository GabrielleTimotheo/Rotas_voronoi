import voronoi
import matplotlib.pyplot as plt
import a_star
import send_mission
import numpy as np
import rospy
import gps_reader

def generate_mission(path):
    for i in range(2): 
        send_mission.generate_mission(path) # lista de tuplas
    return "Missão gerada."

# Test A* algorithm
if __name__ == "__main__":

    # Apply A* algorithm to find the path through the Voronoi Diagram
    astar = a_star.AStarGraph()
    path, voronoi_sites = astar.AStar()

    np.savetxt("path.txt", path, fmt="%s", delimiter=",")

    # Create Voronoi Diagram
    vor = voronoi.VoronoiDiagram()
    vor.PlotVoronoiDiagram()

    # Plot path
    astar.PlotPath(path)

    plt.show()

    print(path)

    # Send mission
    msg = generate_mission(path)
    print(msg)

    # try:
    #     gps = gps_reader.RobotGPSPlotter(path, voronoi_sites)
    #     gps.run()

    #     # Quando a execução do rospy for encerrada, plota os dados
    #     # gps.plot_gps_data()

    # except rospy.ROSInterruptException:
    #     pass
