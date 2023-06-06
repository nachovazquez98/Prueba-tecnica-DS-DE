import os
import requests
import pandas as pd
# direccion de trabajo
cwd = os.getcwd()
path = os.path.dirname(os.path.dirname(cwd))
# path = '/home/nacho/Documents/TEST_DE_PROGRAMACION_JR_DATA_ENGINEER'
os.chdir(path)
# plantilla para guardar serie de tiempo por cada pais
def data_extraction():
    df_pobl = pd.DataFrame(index = pd.date_range('1960', '2022', freq='Y', name='year'), columns = ['mundial'])
    df_pobl.index = df_pobl.index.year
    # definir url y page para saber el limite de pags
    url = "http://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json&page=1"
    page = requests.get(url)
    # webscrappi8ng donde se itera pagina por pagina y año por año por cada pais
    # se almacena en una lista y luego en un dataframe
    last_page = page.json()[0]['pages']
    list_poblacion = []
    for page_number in range(1, last_page + 1):
        page = requests.get("http://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json&page=" + str(page_number))
        print(page_number)
        country_name = page.json()[1][0]['country']['value']
        for i in range(0, len(page.json()[1])):
            value_poblacion = page.json()[1][i]['value']
            list_poblacion.append(value_poblacion)
            if len(list_poblacion) == 62:
                list_poblacion.reverse()
                df_pobl[str(country_name)] = list_poblacion
                list_poblacion.clear()
                pass 
    #se borran los paises que tngan valores nulos
    df_pobl = df_pobl.dropna(axis=1, how='any')
    #se realiza una suma para generar la población mundial
    df_pobl['mundial'] = df_pobl.iloc[:,1:-1].sum(axis=1) 
    #guard archivo en csv
    return df_pobl
    # df_pobl.to_csv("df_pobl.csv")

