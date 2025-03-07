from geographiclib.geodesic import Geodesic
import math
import matplotlib.pyplot as plt
import voronoi
import pandas as pd

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

def PlotEquipamentCoordinates():
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
vertice_REATOR = (3.054527, 1.534546)
vertice_PR = (0.6181564, 0.6871033)
vertice_TPC = (0.7777786, 0.864502)
vertice_IP = (0.5235214, 0.7239075)
vertice_SECH = (0.5302429, 3.303741)
vertice_TC = (0.7042313, 0.8567124)
vertice_SECV = (0.54982, 1.088993)
vertice_DISJUNTOR = (0.7382889, 2.458675)
vertice_BUSCSB = (0.54982, 0.7805481)
vertice_BUSIP = (0.54982, 0.7805519)
vertice_ESTRUTURA2 = [(15.44938, 2.735949), (-12.86117, 2.716169), (12.8493, -2.735949), (-15.46126, -2.716169)]
vertice_ESTRUTURA3 = ()

vertices = [vertice_REATOR, vertice_PR, vertice_TPC, vertice_IP, vertice_SECH, vertice_TC, vertice_SECV, vertice_DISJUNTOR, vertice_BUSCSB, vertice_BUSIP, vertice_ESTRUTURA2, vertice_ESTRUTURA3] 
name = ["REATOR", "PR", "TPC", "IP", "SECH", "TC", "SECV", "DISJUNTOR", "BUSCSB", "BUSIP", "ESTRUTURA2", "ESTRUTURA3"]

for k in range(len(vertices)):
    vertice = vertices[k]

    for i in range(equipment_lat.size):
        
        print(name[k], model_name[i])
        if name[k] in model_name[i]:

            if name[k] == "ESTRUTURA2" or name[k] == "ESTRUTURA3":
                vx = []
                vy = []

                for j in range(0,len(vertice), 2):

                    ponto_medio = (abs(abs(vertice[j][0])-abs(vertice[j+1][0])), abs(abs(vertice[j][1])-abs(vertice[j+1][1])))

                    # Parâmetros iniciais
                    lat0 = ponto_medio[0] # Latitude inicial (ponto médio)
                    lon0 = ponto_medio[1]  # Longitude inicial (ponto médio)
                    alt0 = 0  # Altitude inicial

                    # Deslocamentos em metros (valores de exemplo)
                    x = vertice[j][0]  # deslocamento no eixo X (longitude)
                    y = vertice[j][1]   # deslocamento no eixo Y (latitude)

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

                    # Calcular os vetores
                    vetores_x, vetores_y = calcular_vetores(lat, lon, ponto_medio)

                    # Exibir os vetores calculados
                    print("Vetores X:", vetores_x)
                    print("Vetores Y:", vetores_y)

                    # distancia_x = Geodesic.WGS84.Inverse(ponto_medio[0],ponto_medio[1],ponto_medio[0],lon)['s12']
                    # distancia_y = Geodesic.WGS84.Inverse(ponto_medio[0],ponto_medio[1],lat,ponto_medio[1])['s12']

                    vx.append(vetores_x)
                    vy.append(vetores_y)

                    PlotEquipamentCoordinates()

                # Adicionar os vetores ao dataframe
                df.loc[i, 'Vx'] = str(vx)
                df.loc[i, 'Vy'] = str(vy)

                # Salvar as mudanças no arquivo
                df.to_excel("models_updated.xlsx", index=False)
                print("Arquivo atualizado salvo como 'models_updated.xlsx'")   

            # else:
            ponto_medio = (equipment_lat[i], equipment_lon[i])

            # Parâmetros iniciais
            lat0 = ponto_medio[0] # Latitude inicial (ponto médio)
            lon0 = ponto_medio[1]  # Longitude inicial (ponto médio)
            alt0 = 0  # Altitude inicial

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

            # PlotEquipamentCoordinates()
