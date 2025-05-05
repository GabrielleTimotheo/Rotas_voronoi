from geographiclib.geodesic import Geodesic
import math
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json

class TreatData:
    """ Treat data about equipment dimensions and coordinates.
    """

    def __init__(self):

        self.df = self.LoadFileToDataframe()

        self.equipment_lat = self.df['Latitude']
        self.equipment_lon = self.df['Longitude']
        self.model_name = self.df['Model Name']

        # Equipamentos
        # Metade da largura e altura [metros]
        vertice_REATOR = (1.534546, 3.054527) 
        vertice_PR = (0.6871033, 0.6181564) 
        vertice_TPC = (0.864502, 0.7777786) 
        vertice_IP = (0.7239075, 0.5235214) 
        vertice_SECH = (3.303741, 0.5302429) 
        vertice_TC = (0.8567124, 0.7042313) 
        vertice_SECV = (1.088993, 0.54982) 
        vertice_DISJUNTOR = (2.458675, 0.7382889) 
        vertice_BUSCSB = (0.7805481, 0.54982) 
        vertice_BUSIP = (0.7805519, 0.54982) 
        vertice_ESTRUTURA2 = (2.74, 1.3, -2.74, -1.3)
        vertice_ESTRUTURA3 = (2.74, 1.3, -2.74, -1.3)
        vertice_canaletas_horizontal = (11, 2) # Não é metade, horizontal

        self.vertices_canaletas = {
            "canaleta1": (11, 2, -0.459704, -123.217343),
            "canaleta2": (2, 23.6, -13.222777, -118.792624),
            "canaleta3": (2, 16.62, 8.387840, -118.721732)}

        # Latitude e longitude central de cada canaleta
        # Horizontal
        lat0, lon0 = -3.123199, -41.764537 # Base for the transformation

        # canaleta1 = (-0.459704, -123.217343)
        # canLat1, canLon1 = CartesianToGeodesic(canaleta1[0], canaleta1[1], lat0, lon0)
        # canaleta1 = (canLat1, canLon1)

        # # Vertical
        # canaleta1 = (-0.459704, -123.217343)

        for name, canaleta in self.vertices_canaletas.items():
            novo_lat, novo_lon = self.CartesianToGeodesic(canaleta[2], canaleta[3], lat0, lon0)
            self.vertices_canaletas[name] = (canaleta[0], canaleta[1], novo_lat, novo_lon)

        # Other objects
        self.vertices = [vertice_REATOR, vertice_PR, vertice_TPC, vertice_IP, vertice_SECH, vertice_TC, vertice_SECV, vertice_DISJUNTOR, vertice_BUSCSB, vertice_BUSIP, vertice_ESTRUTURA2, vertice_ESTRUTURA3, vertice_canaletas_horizontal] 
        self.name = ["REATOR", "PR", "TPC", "IP", "SECH", "TC", "SECV", "DISJUNTOR", "BUSCSB", "BUSIP", "ESTRUTURA2", "ESTRUTURA3", "canaletas"]

    def LoadFileToDataframe(self, file_path = "models2.xlsx"):
        """
        Load file to pandas dataframe.

        Args:
            file_path (str): File path
        Returns:
            pd.DataFrame: Dataframe
        """
        try:
            df = pd.read_excel(file_path)
            return df
        except Exception as e:
            print(f"Could not load the file: {e}")
            return None
        
    def calcular_vetores(self, lat, lon, ponto_medio):
        """
        Calculate the vectors x and y from the midpoint.

        Args:
            lat (float): Latitude of the point
            lon (float): Longitude of the point
            ponto_medio (tuple): Midpoint coordinates (lat, lon)
        
        Returns:
            tuple: Vectors x and y
        """
        vetores_x = []
        vetores_y = []
        
        vetores_x =[(ponto_medio[1], ponto_medio[0]), (lon, ponto_medio[0])]
        vetores_y = [(ponto_medio[1], ponto_medio[0]), (ponto_medio[1],lat)]
            
        return vetores_x, vetores_y

    def PlotEquipamentDimensions(self, dx, dy, lat_lon_central):
        """
        Plot equipment coordinates as rectangles with given dimensions (dx and dy in meters),
        centered at (equipment_lat, equipment_lon).

        Args:
            dx (float): Width of the rectangle in meters
            dy (float): Height of the rectangle in meters
            equipment_lat (list): List of equipment latitudes
            equipment_lon (list): List of equipment longitudes
        
        Returns:
            None
        """
        fig, ax = plt.subplots()

        # Converte metros para graus de latitude e longitude
        # 1 grau de latitude = ~111.32 km
        # 1 grau de longitude depende da latitude

        # Plotar os retângulos
        for latlon, w, h in zip(lat_lon_central, dx, dy):

            if all(isinstance(i, float) for i in latlon) and len(latlon) != 0:
                x = latlon[0]
                y = latlon[1]

                w = w / 111320  # Aproximação
                h = h / (40075000 * np.cos(np.radians(x)) / 360)

                ax.scatter(x, y, color='blue')
                rect = patches.Rectangle((x - w/2, y - h/2), w, h,
                                        linewidth=1, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

            # Para o caso das estruturas
            elif all(isinstance(i, list) for i in latlon) and len(latlon) != 0:
                for latlons in latlon:
                    x = latlons[0]
                    y = latlons[1]
                    w1 = w / 111320  # Aproximação
                    h1 = h / (40075000 * np.cos(np.radians(x)) / 360)

                    ax.scatter(x, y, color='blue')
                    rect = patches.Rectangle((x - w1/2, y - h1/2), w1, h1,
                                            linewidth=1, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
            else:
                continue

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Equipments in Geo Coordinates")
        ax.legend(["Equipment Center"])
        ax.grid(True)
        plt.axis('equal')
        plt.show()

    def safe_json_load(self, val):
        """
        Safely load a JSON string into a Python object.
        
        Args:
            val (str): JSON string
        
        Returns:
            dict or None: Loaded JSON object or None if the string is not valid JSON
        """
        if isinstance(val, str):
            return json.loads(val)
        if pd.isna(val):
            return []
        return val 

    def CartesianToGeodesic(self, x, y, lat0, lon0):
        """
        Convert Cartesian coordinates (x, y) to geodesic coordinates (latitude, longitude)
        using the WGS84 ellipsoid model.
        
        Args:
            x (float): X coordinate in meters
            y (float): Y coordinate in meters
            lat0 (float): Initial latitude in degrees
            lon0 (float): Initial longitude in degrees

        Returns:
            tuple: Latitude and longitude in degrees
        """

        # Criar o objeto Geodesic com o modelo WGS84
        geod = Geodesic.WGS84

        # Calcular o azimute e a distância a partir dos deslocamentos cartesianos
        azimuth = math.atan2(y, x) * 180 / math.pi  # Converte o ângulo para graus
        distance = math.sqrt(x**2 + y**2)  # Distância no plano XY

        # Usar o método Direct para calcular a nova latitude e longitude
        result = geod.Direct(lat0, lon0, azimuth, distance)

        # Latitude, longitude e altitude calculados
        lat = result['lat2']
        lon = result['lon2']

        return lat, lon

    def extrair_lat_lon_principal(self, grupos):
        """
        Extract the main latitude and longitude from a list of groups.
        
        Args:
            grupos (list): List of groups containing latitude and longitude
        
        Returns:
            list: List of main latitude and longitude
        """

        if not isinstance(grupos, list):
            print("Não é lista:", grupos)
            return []

        elif len(grupos) == 2:
            resultado = grupos[0]
        else:
            resultado = []
            for subgrupo in grupos:
                resultado.append(subgrupo[0])
        
        return resultado

    def processar_canaleta(self, i):
        i = 0 
        for canaleta, vertice in self.vertices_canaletas.items():
            ponto_medio = (vertice[2], vertice[3])
            x, y = vertice[0], vertice[1]
            lat, lon = self.CartesianToGeodesic(x, y, vertice[0], vertice[1])
            vetores_x, vetores_y = self.calcular_vetores(lat, lon, ponto_medio)

            i = i + 1
            self.df.loc[i, 'Model Name'] = f'ARGO_PARNAIBAIII_V2_LD::BASE::{canaleta}'
            self.df.loc[i, 'LatLonCentral'] = json.dumps([ponto_medio[1], ponto_medio[0]])
            self.df.loc[i, 'Vx'] = json.dumps(vetores_x)
            self.df.loc[i, 'Vy'] = json.dumps(vetores_y)
            self.df.loc[i, 'LarguraMetros'] = x 
            self.df.loc[i, 'ComprimentoMetros'] = y 

    def processar_outros(self, i, vertice):
        ponto_medio = (self.equipment_lat[i], self.equipment_lon[i])
        x, y = vertice
        lat, lon = self.CartesianToGeodesic(x, y, ponto_medio[0], ponto_medio[1])
        vetores_x, vetores_y = self.calcular_vetores(lat, lon, ponto_medio)

        self.df.loc[i, 'LatLonCentral'] = json.dumps([ponto_medio[1], ponto_medio[0]])
        self.df.loc[i, 'Vx'] = json.dumps(vetores_x)
        self.df.loc[i, 'Vy'] = json.dumps(vetores_y)
        self.df.loc[i, 'LarguraMetros'] = x * 2
        self.df.loc[i, 'ComprimentoMetros'] = y * 2

    def processar_estrutura(self, i, vertice):
        vx, vy, ponto_medio_total = [], [], []

        if "ESTRUTURA2" in self.model_name[i]:
            coord = [14.2, 0, -14.2, 0]
        else:
            coord = [28.2, 0, -28.2, 0]
            ponto_medio = (self.equipment_lat[i], self.equipment_lon[i])

            # Parâmetros iniciais
            lat0 = ponto_medio[0] # Latitude inicial (ponto médio)
            lon0 = ponto_medio[1]  # Longitude inicial (ponto médio)

            # Deslocamentos em metros (valores de exemplo)
            x = vertice[0]  # deslocamento no eixo X (longitude)
            y = vertice[1]   # deslocamento no eixo Y (latitude)

            lat, lon = self.CartesianToGeodesic(x, y, lat0, lon0)

            # Calcular os vetores
            vetores_x, vetores_y = self.calcular_vetores(lat, lon, ponto_medio)

            ponto_medio_total.append([ponto_medio[1], ponto_medio[0]])

            vx.append(vetores_x)
            vy.append(vetores_y)

        #------------ Ponto médio para a estrutura como um todo--------------
        ponto_medio_ambos = (self.equipment_lat[i], self.equipment_lon[i])

        # Parâmetros iniciais
        lat0 = ponto_medio_ambos[0] # Latitude inicial (ponto médio)
        lon0 = ponto_medio_ambos[1]  # Longitude inicial (ponto médio)

        # Deslocamentos em metros (valores de exemplo)
        x = coord[0]  # deslocamento no eixo X (longitude)
        y = coord[1]   # deslocamento no eixo Y (latitude)
        
        # Novo ponto médio para cima
        lat1, lon1 = self.CartesianToGeodesic(x, y, lat0, lon0)
        ponto_medio_total.append([lon1, lat1])

        x = coord[2]   # deslocamento no eixo X (longitude)
        y = coord[3] # deslocamento no eixo Y (latitude)
        
        # Novo ponto médio para baixo
        lat2, lon2 = self.CartesianToGeodesic(x, y, lat0, lon0)
        ponto_medio_total.append([lon2, lat2])
        #--------------------------------------------------------------------

        # Ponto médio em cima
        ponto_medio1 = (lat1, lon1)
        # Ponto médio em baixo
        ponto_medio2 = (lat2, lon2)

        # Deslocamentos em metros (valores de exemplo)
        x = vertice[0]  # deslocamento no eixo X (longitude)
        y = vertice[1]   # deslocamento no eixo Y (latitude)

        lat1, lon1 = self.CartesianToGeodesic(x, y, lat1, lon1)

        x = vertice[2]  # deslocamento no eixo X (longitude)
        y = vertice[3]   # deslocamento no eixo Y (latitude)
        lat2, lon2 = self.CartesianToGeodesic(x, y, lat2, lon2)

        # Calcular os vetores
        vetores_x, vetores_y = self.calcular_vetores(lat1, lon1, ponto_medio1)

        vx.append(vetores_x)
        vy.append(vetores_y)

        vetores_x, vetores_y = self.calcular_vetores(lat2, lon2, ponto_medio2)

        vx.append(vetores_x)
        vy.append(vetores_y)

        # Adicionar os vetores ao dataframe
        self.df.loc[i, 'LatLonCentral'] = json.dumps(ponto_medio_total)
        self.df.loc[i, 'Vx'] = json.dumps(vx)
        self.df.loc[i, 'Vy'] = json.dumps(vy)
        self.df.loc[i, 'LarguraMetros'] = vertice[0]*2
        self.df.loc[i, 'ComprimentoMetros'] = vertice[1]*2
    
    def main(self):

        for k in range(len(self.vertices)):
            vertice = self.vertices[k]

            for i in range(self.equipment_lat.size):
                
                if self.name[k] in self.model_name[i]:

                    if "ESTRUTURA2" in self.model_name[i] or "ESTRUTURA3" in self.model_name[i]:
                        self.processar_estrutura(i, vertice)
                        self.df.to_excel("models_updated.xlsx", index=False)

                    elif "canaletas" in self.model_name[i]:
                        self.processar_canaleta(i)
                        self.df.to_excel("models_updated.xlsx", index=False)
                    else:

                        self.processar_outros(i, vertice)
                        self.df.to_excel("models_updated.xlsx", index=False)       

        self.df['LarguraMetros'] = self.df['LarguraMetros'].apply(self.safe_json_load)
        self.df['ComprimentoMetros'] = self.df['ComprimentoMetros'].apply(self.safe_json_load)
        dx = self.df['LarguraMetros']
        dy = self.df['ComprimentoMetros']

        self.df['Vx'] = self.df['Vx'].apply(self.safe_json_load)
        self.df['Vy'] = self.df['Vy'].apply(self.safe_json_load)
        lat_lon_central = self.df['LatLonCentral'].apply(self.safe_json_load)

        self.PlotEquipamentDimensions(dx, dy, lat_lon_central)

if __name__ == "__main__":  

    treat_data = TreatData()
    treat_data.main()