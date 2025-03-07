import numpy as np    

class SmoothPath:    

    def __init__(self):
        self.path = np.loadtxt("path.txt", delimiter=",", dtype=float)
        self.path = np.array(self.path)

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
        
if __name__ == "__main__":

    smooth_path = SmoothPath()
    simplified_path = smooth_path.RamerDouglasPeucker(smooth_path.path, epsilon=0.001)
    print(simplified_path)
    np.savetxt("simplified_path.txt", simplified_path, delimiter=",")