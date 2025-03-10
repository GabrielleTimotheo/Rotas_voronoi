from geographiclib.geodesic import Geodesic
import math
import matplotlib.pyplot as plt
import voronoi
import pandas as pd
import re

def LoadFileToDataframe(file_path = "models.xlsx"):
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
    
# Função para calcular os vetores x e y a partir do ponto médio
def calcular_vetores(lat, lon, ponto_medio):
    vetores_x = []
    vetores_y = []
    

    vetores_x =[(ponto_medio[1], ponto_medio[0]), (lon, ponto_medio[0])]
    vetores_y = [(ponto_medio[1], ponto_medio[0]), (ponto_medio[1],lat)]
        
    return vetores_x, vetores_y

def PlotEquipamentCoordinates(vetores_x, vetores_y):
    """
    Plot equipament coordinates and collect points to the mission.
    """
    fig, ax = plt.subplots()
    ax.scatter(equipment_lon, equipment_lat, picker=True, label='Equipment')
    ax.scatter(vetores_x[1][0], vetores_x[1][1], picker=True, color='red', label='Robot Position')
    ax.scatter(vetores_y[1][0], vetores_y[1][1], picker=True, color='red', label='Robot Position')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.show()

def CartesianToGeodesic(x, y, lat0, lon0):

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
    
df = LoadFileToDataframe()
equipment_lat = df['Latitude']
equipment_lon = df['Longitude']
model_name = df['Model Name']


# Equipamentos
vertice_REATOR = (1.534546, 3.054527) #(3.054527, 1.534546)
vertice_PR = (0.6871033, 0.6181564) #(0.6181564, 0.6871033)
vertice_TPC = (0.864502, 0.7777786) #(0.7777786, 0.864502)
vertice_IP = (0.7239075, 0.5235214) #(0.5235214, 0.7239075)
vertice_SECH = (3.303741, 0.5302429) #(0.5302429, 3.303741)
vertice_TC = (0.8567124, 0.7042313) #(0.7042313, 0.8567124)
vertice_SECV = (1.088993, 0.54982) #(0.54982, 1.088993)
vertice_DISJUNTOR = (2.458675, 0.7382889) #(0.7382889, 2.458675)
vertice_BUSCSB = (0.7805481, 0.54982) #(0.54982, 0.7805481)
vertice_BUSIP = (0.7805519, 0.54982) #(0.54982, 0.7805519)
vertice_ESTRUTURA2 = (1.3, 2.74, -1.3, -2.74)
vertice_ESTRUTURA3 = (1.3, 2.74, -1.3, -2.74)

vertices = [vertice_REATOR, vertice_PR, vertice_TPC, vertice_IP, vertice_SECH, vertice_TC, vertice_SECV, vertice_DISJUNTOR, vertice_BUSCSB, vertice_BUSIP, vertice_ESTRUTURA2, vertice_ESTRUTURA3] 
name = ["REATOR", "PR", "TPC", "IP", "SECH", "TC", "SECV", "DISJUNTOR", "BUSCSB", "BUSIP", "ESTRUTURA2", "ESTRUTURA3"]

for k in range(len(vertices)):
    vertice = vertices[k]

    for i in range(equipment_lat.size):
        
        if name[k] in model_name[i]:
            
            if "ESTRUTURA2" in model_name[i] or "ESTRUTURA3" in model_name[i]:
                vx = []
                vy = []

                if "ESTRUTURA2" in model_name[i]:
                    coord = [14.2, 0, -14.2, 0]
                else:
                    coord = [28.2, 0, -28.2, 0]
                    ponto_medio = (equipment_lat[i], equipment_lon[i])

                    # Parâmetros iniciais
                    lat0 = ponto_medio[0] # Latitude inicial (ponto médio)
                    lon0 = ponto_medio[1]  # Longitude inicial (ponto médio)

                    # Deslocamentos em metros (valores de exemplo)
                    x = vertice[0]  # deslocamento no eixo X (longitude)
                    y = vertice[1]   # deslocamento no eixo Y (latitude)

                    lat, lon = CartesianToGeodesic(x, y, lat0, lon0)

                    # Calcular os vetores
                    vetores_x, vetores_y = calcular_vetores(lat, lon, ponto_medio)

                    vx.append(vetores_x)
                    vy.append(vetores_y)

                #------------ Ponto médio para a estrutura como um todo--------------
                ponto_medio_ambos = (equipment_lat[i], equipment_lon[i])

                # Parâmetros iniciais
                lat0 = ponto_medio_ambos[0] # Latitude inicial (ponto médio)
                lon0 = ponto_medio_ambos[1]  # Longitude inicial (ponto médio)

                # Deslocamentos em metros (valores de exemplo)
                x = coord[0]  # deslocamento no eixo X (longitude)
                y = coord[1]   # deslocamento no eixo Y (latitude)
                
                # Novo ponto médio para cima
                lat1, lon1 = CartesianToGeodesic(x, y, lat0, lon0)

                x = coord[2]   # deslocamento no eixo X (longitude)
                y = coord[3] # deslocamento no eixo Y (latitude)
                
                # Novo ponto médio para baixo
                lat2, lon2 = CartesianToGeodesic(x, y, lat0, lon0)
                #--------------------------------------------------------------------

                # Ponto médio em cima
                ponto_medio1 = (lat1, lon1)
                # Ponto médio em baixo
                ponto_medio2 = (lat2, lon2)

                # Deslocamentos em metros (valores de exemplo)
                x = vertice[0]  # deslocamento no eixo X (longitude)
                y = vertice[1]   # deslocamento no eixo Y (latitude)

                lat1, lon1 = CartesianToGeodesic(x, y, lat1, lon1)

                x = vertice[2]  # deslocamento no eixo X (longitude)
                y = vertice[3]   # deslocamento no eixo Y (latitude)
                lat2, lon2 = CartesianToGeodesic(x, y, lat2, lon2)

                # Calcular os vetores
                vetores_x, vetores_y = calcular_vetores(lat1, lon1, ponto_medio1)

                vx.append(vetores_x)
                vy.append(vetores_y)

                vetores_x, vetores_y = calcular_vetores(lat2, lon2, ponto_medio2)

                vx.append(vetores_x)
                vy.append(vetores_y)

                # PlotEquipamentCoordinates(vetores_x, vetores_y)

                # Adicionar os vetores ao dataframe
                df.loc[i, 'Vx'] = str(vx)
                df.loc[i, 'Vy'] = str(vy)

                # Salvar as mudanças no arquivo
                df.to_excel("models_updated.xlsx", index=False)
                print("Arquivo atualizado salvo como 'models_updated.xlsx'")   

            else:
                ponto_medio = (equipment_lat[i], equipment_lon[i])

                # Parâmetros iniciais
                lat0 = ponto_medio[0] # Latitude inicial (ponto médio)
                lon0 = ponto_medio[1]  # Longitude inicial (ponto médio)

                # Deslocamentos em metros (valores de exemplo)
                x = vertice[0]  # deslocamento no eixo X (longitude)
                y = vertice[1]   # deslocamento no eixo Y (latitude)

                lat, lon = CartesianToGeodesic(x, y, lat0, lon0)

                # Calcular os vetores
                vetores_x, vetores_y = calcular_vetores(lat, lon, ponto_medio)

                # Exibir os vetores calculados
                print("Vetores X:", vetores_x)
                print("Vetores Y:", vetores_y)

                distancia_x = Geodesic.WGS84.Inverse(ponto_medio[0],ponto_medio[1],ponto_medio[0],lon)['s12']
                distancia_y = Geodesic.WGS84.Inverse(ponto_medio[0],ponto_medio[1],lat,ponto_medio[1])['s12']

                # Adicionar os vetores ao dataframe
                df.loc[i, 'Vx'] = str(vetores_x)
                df.loc[i, 'Vy'] = str(vetores_y)

                # Salvar as mudanças no arquivo
                df.to_excel("models_updated.xlsx", index=False)
                print("Arquivo atualizado salvo como 'models_updated.xlsx'")

                # PlotEquipamentCoordinates(vetores_x, vetores_y)
