import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import seaborn as sns; sns.set()
from darts import TimeSeries
from sklearn.svm import SVR
from darts.metrics import mape, mse, mae 
from darts.models import Prophet
from darts.models import (
    ARIMA,
    AutoARIMA,
    ExponentialSmoothing,
    FFT
)

df_pobl = data_extraction()
# df_pobl = pd.read_csv("df_pobl.csv", index_col=0)
#se pueden agregar mas modelos adentro de la lista
ts_models = [
    # ARIMA(), 
    # AutoARIMA(),
    FFT(),
    # ExponentialSmoothing(),
    Prophet()
    ]
def ts_prediction(df, split, date_range):
        df_pred = dataset.copy()
        df_pred = pd.DataFrame(df_pred)
        df_pred['Year'] = pd.date_range('1960', '2022', freq='Y')
        series = TimeSeries.from_dataframe(df_pred, 'Year', dataset.name)
        train, val = series.split_before(pd.Timestamp('20181231'))
        print(dataset.name)
        forecaster.fit(train)
        prediction = forecaster.predict(len(val))
        return prediction
####
df_forecasts = pd.DataFrame(index = pd.date_range('1960', '2026', freq='Y', name='year'), columns = ['test'])
df_forecasts.index = df_forecasts.index.year
####
#iterar sobre cada columna del dataset
for column in df_pobl:
    dataset = df_pobl[column]
    # se realiza el entrenamiento y evaluacion incicial, se calculan las metricas 
    # y se guard la informacion de cada modelo adenmtro de una lista con diccionarios
    dict_list = ['name', 'prediction', 'model', 'mape', 'mae', 'mse']
    class_dict = {}
    class_list = []

    for i, forecaster in enumerate(ts_models):
        class_dict = {}
        class_dict[dict_list[0]] = (forecaster.__class__.__name__)
        #############
        prediction = ts_prediction(dataset, 20181231, '1960', '2022')
        ##############
        class_dict[dict_list[1]] = prediction
        class_dict[dict_list[2]] = forecaster
        class_dict[dict_list[3]] = mape(val, prediction)
        class_dict[dict_list[4]] = mae(val, prediction)
        class_dict[dict_list[5]] = mse(val, prediction)
        
        class_list.append(class_dict)
    #se realiza la evalucacion del mejor modelo con la metrica MAPE
    #y retorna un diccionario con el mejor modelo y su informacion
    minl = []
    for dicts in class_list:
        print(dicts['name'],':',dicts['mape'])
        minl.append(dicts['mape'])

    min_mape = min(minl)

    def return_best_mape(class_list, min_mape):
        for dicts in class_list:
            if dicts['mape'] == min_mape:
                return dicts

    best_mape = return_best_mape(class_list, min_mape)
    print(best_mape)
    ####
    #se realiza la prediccion con los siguientes 4 años
    best_prediction = best_mape['model'].predict(8)
    df_prediction = best_prediction.pd_dataframe()
    #se realiza la grafica con los datos y las predicciones
    fig, ax = plt.subplots() 
    fig.set_size_inches(18, 10)
    ax.plot(dataset.index, dataset.values, linestyle = ':', color = 'black', linewidth=2, label='Población')
    ax.plot(df_prediction.index.year, df_prediction.values, label = 'Predicción', color = 'green')
    plt.title('Población de ' + dataset.name + '\n' + best_mape['name'] + '\nmape: ' + str(best_mape['mape']) + '\nmae: ' + str(best_mape['mae']) + '\nmse: ' + str(best_mape['mse']))
    ax.legend()
    plt.savefig("plots/"+dataset.name+".png")
    print(best_mape['name'], '\nmape: ', best_mape['mape'], '\nmae: ', best_mape['mae'], '\nmse: ', best_mape['mse'])
    ####se concatena la predixxion con los datos de poblaxion en un miso dataframe
    df_prediction.index = df_prediction.index.year
    ts_complete = pd.DataFrame(dataset).append(df_prediction.iloc[-4:])
    ts_complete.rename(columns={str(dataset.name):str(dataset.name) + '_' + str(best_mape['name'])}, inplace=True)
    #se concatena el forecast
    df_forecasts[str(dataset.name) + '_' + str(best_mape['name'])] = ts_complete
    #guardar metricas
    with open('errors.csv', 'a') as errors_file:
        errors_file.write(str(dataset.name) + '_' + str(class_list[0]['name']) + ',' + 'mape_' + str(class_list[0]['mape']) + ',' + 'mae_' + str(class_list[0]['mae']) + ',' + 'mse_' + str(class_list[0]['mse']) + '\n')
        errors_file.write(str(dataset.name) + '_' + str(class_list[1]['name']) + ',' + 'mape_' + str(class_list[1]['mape']) + ',' + 'mae_' + str(class_list[1]['mae']) + ',' + 'mse_' + str(class_list[1]['mse']) + '\n')
        # errors_file.write(str(dataset.name) + '_' + str(class_list[2]['name']) + ',' + 'mape_' + str(class_list[2]['mape']) + ',' + 'mae_' + str(class_list[2]['mae']) + ',' + 'mse_' + str(class_list[2]['mse']) + '\n')

df_forecasts.to_csv("forecasts.csv")
