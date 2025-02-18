import numpy as np
import matplotlib.pyplot as plt

def read_selected_points():
    """
    Read selected points from a text file.

    Args:
        file_path (str): Path to the text file.

    Returns:
        np.ndarray: Array of selected points.
    """
    try:
        points = np.loadtxt("Rotas_voronoi/rotas/selected_points.txt", delimiter=",", dtype=float)
        points = np.array(points)
        return points
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def AddExtraPoints(points):
    """
    Add extra points to the selected points.

    Args:
        points (np.ndarray): Array of selected points.

    Returns:
        np.ndarray: Array of selected points with extra points.
    """
    # Fator de afastamento
    k = 0.5  

    # Calcula o centroide
    centro = np.mean(points, axis=0)

    # Calcula os novos pontos afastados
    new_points = []
    for p in points:
        direction = p - centro
        new_point = p + k * direction
        new_points.append(new_point)

    new_points = np.array(new_points)
    return new_points, centro

selected_points = read_selected_points()
new_points, centro = AddExtraPoints(selected_points)

# Plotando os pontos antes e depois
plt.scatter(selected_points[:,0], selected_points[:,1], color='blue', label='Original')
plt.scatter(new_points[:,0], new_points[:,1], color='red', label='Afastado')
plt.scatter(centro[0], centro[1], color='black', marker='x', label='Centroide')
plt.legend()
plt.grid()
plt.show()