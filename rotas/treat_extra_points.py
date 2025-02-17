import numpy as np

def read_selected_points(file_path):
    """
    Read selected points from a text file.

    Args:
        file_path (str): Path to the text file.

    Returns:
        np.ndarray: Array of selected points.
    """
    try:
        points = np.loadtxt(file_path, delimiter=",")
        return points
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

file_path = "selected_points.txt"
selected_points = read_selected_points(file_path)

if selected_points is not None:
    print("Selected points:")
    print(selected_points)
else:
    print("No points were read.")