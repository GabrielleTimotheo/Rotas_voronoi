import quadtree
import matplotlib.pyplot as plt
import numpy as np

def OnClick(event):
    """
    Function to be called when a point is clicked.
    
    Args:
        event (matplotlib.backend_bases.MouseEvent): Event object.
    """
    selected_points = []
    if event.xdata is None or event.ydata is None:
        return
    
    click_point = np.array([event.xdata, event.ydata])
    distances = np.linalg.norm(vertices - click_point, axis=1)
    nearest_vertex = vertices[np.argmin(distances)]
    
    selected_points.append(nearest_vertex)
    ax.scatter(nearest_vertex[0], nearest_vertex[1], color='blue', marker='x')
    plt.draw()
    
    with open("selected_points.txt", "a") as f:
        f.write(f"{nearest_vertex[0]}, {nearest_vertex[1]}\n")

if __name__ == "__main__":

    # Clean selected points file
    open("selected_points.txt", "w").close()

    # Load file to dataframe and filter it
    df = quadtree.LoadFileToDataframe()
    points_to_use_quadtree = quadtree.FilterDataframe(df)
    
    # Min and max latitude and longitude
    min_lat = points_to_use_quadtree[:, 1].min()
    max_lat = points_to_use_quadtree[:, 1].max()
    min_lon = points_to_use_quadtree[:, 0].min()
    max_lon = points_to_use_quadtree[:, 0].max()

    # Create quadtree
    boundary = (min_lon, min_lat, max_lon, max_lat) # Area boundary

    quad = quadtree.Quadtree(boundary)

    for point in points_to_use_quadtree:
        quad.Insert(point)

    # Plot
    fig, ax = plt.subplots()
    quad.Plot(ax)
    ax.scatter(points_to_use_quadtree[:, 0], points_to_use_quadtree[:, 1], color='red')
    ax.set_aspect('equal')

    # Interactive point selection
    vertices = np.array(quad.GetVertices())

    fig.canvas.mpl_connect('button_press_event', OnClick)
    plt.show()