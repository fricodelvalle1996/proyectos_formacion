'''
OJO: algunos fragmentos del código se han ejecutado en desorden
'''



### PARTE 1: DEPURACIÓN ###


# Cargo las librerias 
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Establecemos nuestro escritorio de trabajo
os.chdir('C:\\Users\\frico\\OneDrive\\Escritorio\\MÁSTER\\07. MINERÍA DE DATOS\\Práctica')

# Cargo las funciones que voy a utilizar
from FuncionesMineria import (analizar_variables_categoricas, cuentaDistintos, frec_variables_num, 
                           atipicosAmissing, patron_perdidos, ImputacionCuant, ImputacionCuali)

# Cargo los datos
datos = pd.read_excel('DatosEleccionesEspaña.xlsx')


# Comprobamos el tipo de formato de las variables variable que se ha asignado en la lectura.
datos.dtypes

# Indico las categóricas que aparecen como numéricas
numericasAcategoricas = ['AbstencionAlta', 'Izquierda', 'Derecha']

# Las transformo en categóricas
for var in numericasAcategoricas:
    datos[var] = datos[var].astype(str)

# Genera una lista con los nombres de las variables.
variables = list(datos.columns)  

# Seleccionar las columnas numéricas del DataFrame
numericas = datos.select_dtypes(include=['int', 'int32', 'int64','float', 'float32', 'float64']).columns

# Seleccionar las columnas categóricas del DataFrame
categoricas = [variable for variable in variables if variable not in numericas]
 
# Comprobamos que todas las variables tienen el formato que queremos  
datos.dtypes

# Frecuencias de los valores en las variables categóricas
analizar_variables_categoricas(datos)

# Si quisieramos conocer las diferentes categorias de una variable
# categórica, por ejemplo CalifProductor
datos['Densidad'].unique()
datos['ActividadPpal'].unique()

# Cuenta el número de valores distintos de cada una de las variables numéricas de un DataFrame
cuentaDistintos(datos)

# Descriptivos variables numéricas mediante función describe() de Python
descriptivos_num = datos.describe().T

# Añadimos más descriptivos a los anteriores
for num in numericas:
    descriptivos_num.loc[num, "Asimetria"] = datos[num].skew()
    descriptivos_num.loc[num, "Kurtosis"] = datos[num].kurtosis()
    descriptivos_num.loc[num, "Rango"] = np.ptp(datos[num].dropna().values)


# Muestra valores perdidos
datos[variables].isna().sum()


# Corregimos los errores detectados

# A veces los 'nan' vienen como como una cadena de caracteres, los modificamos a perdidos (nan).
for x in categoricas:
    datos[x] = datos[x].replace('nan', np.nan) 

# Missings no declarados variables cualitativas (NSNC, ?)
datos['Densidad'] = datos['Densidad'].replace('?', np.nan)

# Missings no declarados variables cuantitativas (-1, 99999)
datos['Explotaciones'] = datos['Explotaciones'].replace(99999, np.nan)

# Valores fuera de rango
datos['ForeignersPtge'] = [x if 0 <= x <= 100 else np.nan for x in datos['ForeignersPtge']]

# Junto categorías poco representadas de las variables categóricas
datos['ActividadPpal'] = datos['ActividadPpal'].replace({'Construccion': 'ConstruccionIndustriaServicios', 
                                                         'Industria': 'ConstruccionIndustriaServicios', 
                                                         'Servicios': 'ConstruccionIndustriaServicios'})
datos['CCAA'] = datos['CCAA'].replace({'PaísVasco': 'NavarraPV', 'Navarra': 'NavarraPV', 'Galicia': 'GaliciaAsturiasCantabria', 
                                       'Asturias': 'GaliciaAsturiasCantabria', 'Cantabria': 'GaliciaAsturiasCantabria', 
                                       'Madrid': 'MadridRiojaCanarias', 'Rioja': 'MadridRiojaCanarias', 'Canarias': 
                                           'MadridRiojaCanarias', 'Extremadura': 'ExtremaduraOtros', 'Baleares': 'ExtremaduraOtros', 
                                           'Murcia': 'ExtremaduraOtros', 'Ceuta': 'ExtremaduraOtros', 'Melilla': 'ExtremaduraOtros'})

# Indico la variableObj, el ID y las Input (los atipicos y los missings se gestionan
# solo de las variables input)
datos = datos.set_index(datos['Name']).drop('Name', axis = 1)
varObjCont = datos['AbstentionPtge']
varObjBin = datos['Derecha']
datos_input = datos.drop(['AbstentionPtge', 'Izda_Pct', 'Dcha_Pct', 'Otros_Pct', 'AbstencionAlta', 
                          'Izquierda', 'Derecha'], axis = 1)

# Genera una lista con los nombres de las variables del cojunto de datos input.
variables_input = list(datos_input.columns)  

# Selecionamos las variables numéricas
numericas_input = datos_input.select_dtypes(include = ['int', 'int32', 'int64','float', 'float32', 'float64']).columns

# Selecionamos las variables categóricas
categoricas_input = [variable for variable in variables_input if variable not in numericas_input]


## ATIPICOS

# Cuento el porcentaje de atipicos de cada variable. 


# La proporción de valores atípicos se calcula dividiendo la cantidad de valores atípicos por el número total de filas
resultados = {x: atipicosAmissing(datos_input[x])[1] / len(datos_input) for x in numericas_input}

# Modifico los atipicos como missings
for x in numericas_input:
    datos_input[x] = atipicosAmissing(datos_input[x])[0]

# MISSINGS
# Visualiza un mapa de calor que muestra la matriz de correlación de valores ausentes en el conjunto de datos.

patron_perdidos(datos_input)
 
    

# Muestra total de valores perdidos por cada variable
datos_input[variables_input].isna().sum()

# Muestra proporción de valores perdidos por cada variable (guardo la información)
prop_missingsVars = datos_input.isna().sum()/len(datos_input)
#Ninguna de las variables tiene más del 50% de valores perdidos, por lo que no eliminamos ninguna (el máximo no llega a 10%)

# Creamos la variable prop_missings que recoge el número de valores perdidos por cada observación
datos_input['prop_missings'] = datos_input.isna().mean(axis = 1)

# Realizamos un estudio descriptivo básico a la nueva variable
datos_input['prop_missings'].describe()
#En este caso tampoco aparece ninguna variable con >50% de valores perdidos, ya que el máximo es de 33%

# Calculamos el número de valores distintos que tiene la nueva variable
len(datos_input['prop_missings'].unique())

# Elimino las observaciones con mas de la mitad de datos missings (no hay ninguna)
eliminar = datos_input['prop_missings'] > 0.5
datos_input = datos_input[~eliminar]
varObjBin = varObjBin[~eliminar]
varObjCont = varObjCont[~eliminar]

# Transformo la nueva variable en categórica (ya que tiene pocos valores diferentes, en este caso 8)
datos_input["prop_missings"] = datos_input["prop_missings"].astype(str)

# Agrego 'prop_missings' a la lista de nombres de variables input
variables_input.append('prop_missings')
categoricas_input.append('prop_missings')


# Elimino las variables con mas de la mitad de datos missings (no hay ninguna)
eliminar = [prop_missingsVars.index[x] for x in range(len(prop_missingsVars)) if prop_missingsVars[x] > 0.5]
datos_input = datos_input.drop(eliminar, axis = 1)


#Recategorizo categoricas con "suficientes" observaciones missings.
'''
La variable con mayor % de missings es Population y no llega al 10%
Se decide por tanto no recategorizar ninguna.
'''
#datos_input['Population'] = datos_input['Population'].fillna('Desconocido')

## IMPUTACIONES
# Imputo todas las cuantitativas
for x in numericas_input:
    datos_input[x] = ImputacionCuant(datos_input[x], 'aleatorio')

# Imputo todas las cualitativas
for x in categoricas_input:
    datos_input[x] = ImputacionCuali(datos_input[x], 'aleatorio')

# Reviso que no queden datos missings
datos_input.isna().sum()



# Una vez finalizado este proceso, se puede considerar que los datos estan depurados. Los guardamos
DatosEleccionesEspana = pd.concat([varObjCont, varObjBin, datos_input], axis = 1)
with open('DatosEleccionesEspana.pickle', 'wb') as archivo:
    pickle.dump(DatosEleccionesEspana, archivo)
    






### SELECCIÓN DE VARIABLES
# Cargo las librerias 
import os
import pickle
from sklearn.model_selection import train_test_split
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


# Cargo las funciones que voy a utilizar
from FuncionesMineria import (Rsq, lm, lm_forward, lm_backward, lm_stepwise, validacion_cruzada_lm,
                           crear_data_modelo)

# Cargo los datos depurados
with open('todo_contE.pickle', 'rb') as f:
    todo = pickle.load(f)
    
# Parto de los datos ya depurados del apartado anterior almacenados en el pickle
with open('DatosEleccionesEspana.pickle', 'rb') as f:
    datosRL = pickle.load(f)

# Identifico la variable objetivo y la elimino del conjunto de datos
varObjCont = datosRL['AbstentionPtge']
#todo = todo.drop('Beneficio', axis = 1)

# Identifico las variables continuas
var_cont = ['Age_under19_Ptge', 'Age_over65_pct', 
                               'Construccion', 'ComercTTEHosteleria','totalEmpresas', 'Industria',
                               'Servicios','Pob2010', 'xAge_under19_Ptge', 'sqrtxAge_over65_pct', 
                                'logxConstruccion', 'logxComercTTEHosteleria','logxtotalEmpresas', 'logxIndustria',
                                 'logxServicios','logxPob2010']



# Identifico las variables continuas sin transformar
var_cont_sin_transf = ['Age_under19_Ptge', 'Age_over65_pct', 
                               'Construccion', 'ComercTTEHosteleria','totalEmpresas', 'Industria',
                               'Servicios','Pob2010']

# Identifico las variables categóricas
var_categ = ['CCAA', 'ActividadPpal']

# Hago la particion
x_train, x_test, y_train, y_test = train_test_split(todo, varObjCont, test_size = 0.2, random_state = 1234567)



# Seleccion de variables Stepwise, métrica AIC
modeloStepAIC = lm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ, [], 'AIC')
# Resumen del modelo
modeloStepAIC['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepAIC['Modelo'], y_train, modeloStepAIC['X'])

# Preparo datos test
x_test_modeloStepAIC = crear_data_modelo(x_test, modeloStepAIC['Variables']['cont'], 
                                                modeloStepAIC['Variables']['categ'], 
                                                modeloStepAIC['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepAIC['Modelo'], y_test, x_test_modeloStepAIC)

# Seleccion de variables Backward, métrica AIC
modeloBackAIC = lm_backward(y_train, x_train, var_cont_sin_transf, var_categ, [], 'AIC')
modeloBackAIC['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloBackAIC['Modelo'], y_train, modeloBackAIC['X'])


x_test_modeloBackAIC = crear_data_modelo(x_test, modeloBackAIC['Variables']['cont'], 
                                                modeloBackAIC['Variables']['categ'], 
                                                modeloBackAIC['Variables']['inter'])


Rsq(modeloBackAIC['Modelo'], y_test, x_test_modeloBackAIC)

# Comparo número de parámetros (iguales)
len(modeloStepAIC['Modelo'].params)
len(modeloBackAIC['Modelo'].params)


# Mismas variables seleccionadas, mismos parámetros, mismo modelo.
modeloStepAIC['Modelo'].params
modeloBackAIC['Modelo'].params

# Seleccion de variables Stepwise, métrica BIC
modeloStepBIC = lm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ, [], 'BIC')
# Resumen del modelo
modeloStepBIC['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepBIC['Modelo'], y_train, modeloStepBIC['X'])

# Preparo datos test
x_test_modeloStepBIC = crear_data_modelo(x_test, modeloStepBIC['Variables']['cont'], 
                                                modeloStepBIC['Variables']['categ'], 
                                                modeloStepBIC['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepBIC['Modelo'], y_test, x_test_modeloStepBIC)

# Seleccion de variables Backward, métrica BIC
modeloBackBIC = lm_backward(y_train, x_train, var_cont_sin_transf, var_categ, [], 'BIC')
# Resumen del modelo
modeloBackBIC['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloBackBIC['Modelo'], y_train, modeloBackBIC['X'])

# Preparo datos test
x_test_modeloBackBIC = crear_data_modelo(x_test, modeloBackBIC['Variables']['cont'], 
                                                modeloBackBIC['Variables']['categ'], 
                                                modeloBackBIC['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloBackBIC['Modelo'], y_test, x_test_modeloBackBIC)

# Comparo número de parámetros
len(modeloBackBIC['Modelo'].params)
len(modeloStepBIC['Modelo'].params)


modeloStepBIC['Modelo'].params
modeloBackBIC['Modelo'].params


# Comparo (R-squared)
Rsq(modeloStepAIC['Modelo'], y_train, modeloStepAIC['X'])
Rsq(modeloStepBIC['Modelo'], y_train, modeloStepBIC['X'])

Rsq(modeloStepAIC['Modelo'], y_test, x_test_modeloStepAIC)
Rsq(modeloStepBIC['Modelo'], y_test, x_test_modeloStepBIC)


# Interacciones 2 a 2 de todas las variables (excepto las continuas transformadas)
interacciones = var_cont_sin_transf + var_categ
interacciones_unicas = list(itertools.combinations(interacciones, 2)) 
  
# Seleccion de variables Stepwise, métrica AIC, con interacciones
modeloStepAIC_int = lm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ, 
                                interacciones_unicas, 'AIC')
# Resumen del modelo
modeloStepAIC_int['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepAIC_int['Modelo'], y_train, modeloStepAIC_int['X'])

# Preparo datos test
x_test_modeloStepAIC_int = crear_data_modelo(x_test, modeloStepAIC_int['Variables']['cont'], 
                                                    modeloStepAIC_int['Variables']['categ'], 
                                                    modeloStepAIC_int['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepAIC_int['Modelo'], y_test, x_test_modeloStepAIC_int)

# Seleccion de variables Stepwise, métrica BIC, con interacciones
modeloStepBIC_int = lm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ,
                                interacciones_unicas, 'BIC')
# Resumen del modelo
modeloStepBIC_int['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepBIC_int['Modelo'], y_train, modeloStepBIC_int['X'])

# Preparo datos test
x_test_modeloStepBIC_int = crear_data_modelo(x_test, modeloStepBIC_int['Variables']['cont'], 
                                                    modeloStepBIC_int['Variables']['categ'], 
                                                    modeloStepBIC_int['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepBIC_int['Modelo'], y_test, x_test_modeloStepBIC_int)
  


# Comparo los R^2 del modelo utilizando ambos criterios
Rsq(modeloStepAIC_int['Modelo'], y_train, modeloStepAIC_int['X'])
Rsq(modeloStepAIC_int['Modelo'], y_test, x_test_modeloStepAIC_int)


Rsq(modeloStepBIC_int['Modelo'], y_train, modeloStepBIC_int['X'])
Rsq(modeloStepBIC_int['Modelo'], y_test, x_test_modeloStepBIC_int)



# Comparo número de parámetros  
len(modeloStepAIC_int['Modelo'].params)
len(modeloStepBIC_int['Modelo'].params)


# Pruebo con todas las transf y las variables originales, métrica AIC
modeloStepAIC_trans = lm_stepwise(y_train, x_train, var_cont, var_categ, [], 'AIC')
# Resumen del modelo
modeloStepAIC_trans['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepAIC_trans['Modelo'], y_train, modeloStepAIC_trans['X'])

# Preparo datos test
x_test_modeloStepAIC_trans = crear_data_modelo(x_test, modeloStepAIC_trans['Variables']['cont'], 
                                                      modeloStepAIC_trans['Variables']['categ'], 
                                                      modeloStepAIC_trans['Variables']['inter'])

# R-squared del modelo para test
Rsq(modeloStepAIC_trans['Modelo'], y_test, x_test_modeloStepAIC_trans)

# Pruebo con todas las transf y las variables originales, métrica BIC
modeloStepBIC_trans = lm_stepwise(y_train, x_train, var_cont, var_categ, [], 'BIC')
# Resumen del modelo
modeloStepBIC_trans['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepBIC_trans['Modelo'], y_train, modeloStepBIC_trans['X'])

# Preparo datos test
x_test_modeloStepBIC_trans = crear_data_modelo(x_test, modeloStepBIC_trans['Variables']['cont'], 
                                                      modeloStepBIC_trans['Variables']['categ'], 
                                                      modeloStepBIC_trans['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepBIC_trans['Modelo'], y_test, x_test_modeloStepBIC_trans)


# Comparo los R^2 de los modelos utilizando ambos criterios
Rsq(modeloStepAIC_trans['Modelo'], y_train, modeloStepAIC_trans['X'])
Rsq(modeloStepAIC_trans['Modelo'], y_test, x_test_modeloStepAIC_trans)


Rsq(modeloStepBIC_trans['Modelo'], y_train, modeloStepBIC_trans['X'])
Rsq(modeloStepBIC_trans['Modelo'], y_test, x_test_modeloStepBIC_trans)


# Comparo número de parámetros  
len(modeloStepAIC_trans['Modelo'].params)
len(modeloStepBIC_trans['Modelo'].params)

# Pruebo modelo con las Transformaciones y las interacciones, métrica AIC
modeloStepAIC_transInt = lm_stepwise(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'AIC')
# Resumen del modelo
modeloStepAIC_transInt['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepAIC_transInt['Modelo'], y_train, modeloStepAIC_transInt['X'])

# Preparo datos test
x_test_modeloStepAIC_transInt = crear_data_modelo(x_test, modeloStepAIC_transInt['Variables']['cont'], 
                                                         modeloStepAIC_transInt['Variables']['categ'], 
                                                         modeloStepAIC_transInt['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepAIC_transInt['Modelo'], y_test, x_test_modeloStepAIC_transInt)

# Pruebo modelo con las Transformaciones y las interacciones, métrica BIC
modeloStepBIC_transInt = lm_stepwise(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'BIC')
# Resumen del modelo
modeloStepBIC_transInt['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepBIC_transInt['Modelo'], y_train, modeloStepBIC_transInt['X'])


# Preparo datos test
x_test_modeloStepBIC_transInt = crear_data_modelo(x_test, modeloStepBIC_transInt['Variables']['cont'], 
                                                         modeloStepBIC_transInt['Variables']['categ'], 
                                                         modeloStepBIC_transInt['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepBIC_transInt['Modelo'], y_test, x_test_modeloStepBIC_transInt)


# Comparo los R^2 de los modelos utilizando ambos criterios
Rsq(modeloStepAIC_transInt['Modelo'], y_train, modeloStepAIC_transInt['X'])
Rsq(modeloStepAIC_transInt['Modelo'], y_test, x_test_modeloStepAIC_transInt)


Rsq(modeloStepBIC_transInt['Modelo'], y_train, modeloStepBIC_transInt['X'])
Rsq(modeloStepBIC_transInt['Modelo'], y_test, x_test_modeloStepBIC_transInt)


# Comparo número de parámetros  
len(modeloStepAIC_transInt['Modelo'].params)
len(modeloStepBIC_transInt['Modelo'].params)


# Hago validacion cruzada repetida para ver que modelo es mejor
# Crea un DataFrame vacío para almacenar resultados
results = pd.DataFrame({
    'Rsquared': []
    , 'Resample': []
    , 'Modelo': []
})
# Realiza el siguiente proceso 20 veces (representado por el bucle `for rep in range(20)`)

for rep in range(20):
    # Realiza validación cruzada en modelos diferentes y almacena sus R-squared en listas separadas

    modelo_stepBIC = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC['Variables']['cont']
        , modeloStepBIC['Variables']['categ']
    )
    modelo_BackBIC = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloBackBIC['Variables']['cont']
        , modeloBackBIC['Variables']['categ']
    )
    modelo_stepBIC_int = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC_int['Variables']['cont']
        , modeloStepBIC_int['Variables']['categ']
        , modeloStepBIC_int['Variables']['inter']
    )
    modelo_stepBIC_trans = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC_trans['Variables']['cont']
        , modeloStepBIC_trans['Variables']['categ']
    )
    modelo_stepBIC_transInt = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC_transInt['Variables']['cont']
        , modeloStepBIC_transInt['Variables']['categ']
        , modeloStepBIC_transInt['Variables']['inter']
    )
    # Crea un DataFrame con los resultados de validación cruzada para esta repetición

    results_rep = pd.DataFrame({
        'Rsquared': modelo_stepBIC + modelo_stepBIC_int + modelo_stepBIC_trans + modelo_stepBIC_trans + modelo_stepBIC_transInt
        , 'Resample': ['Rep' + str((rep + 1))]*5*5 # Etiqueta de repetición (5 repeticiones 6 modelos)
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 + [4]*5 + [5]*5  # Etiqueta de modelo (6 modelos 5 repeticiones)
    })
    results = pd.concat([results, results_rep], axis = 0)
    
# Boxplot de la validacion cruzada 
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráficoç
# Agrupa los valores de Rsquared por modelo
grupo_metrica = results.groupby('Modelo')['Rsquared']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('Rsquared')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico  

# Calcular la media de las métricas R-squared por modelo
media_r2 = results.groupby('Modelo')['Rsquared'].mean()
print(media_r2)
# Calcular la desviación estándar de las métricas R-squared por modelo
std_r2 = results.groupby('Modelo')['Rsquared'].std()
print(std_r2)
# Contar el número de parámetros en cada modelo
num_params = [len(modeloStepAIC['Modelo'].params), len(modeloStepBIC_int['Modelo'].params), 
 len(modeloStepAIC_trans['Modelo'].params), len(modeloStepBIC_trans['Modelo'].params), 
 len(modeloStepBIC_transInt['Modelo'].params)]

print(num_params)


## Seleccion aleatoria (se coge la submuestra de los datos de entrenamiento)
# Concretamente el 70% de los datos de entrenamiento utilizados para contruir los 
# modelos anteriores.
# El método de selección usado ha sido el Backward con el criterio BIC
# Se aplica este método a 30 submuestras diferentes

# Inicializar un diccionario para almacenar las fórmulas y variables seleccionadas.
variables_seleccionadas = {
    'Formula': [],
    'Variables': []
}

# Realizar 30 iteraciones de selección aleatoria.
for x in range(30):
    print('---------------------------- iter: ' + str(x))
    
    # Dividir los datos de entrenamiento en conjuntos de entrenamiento y prueba.
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train, 
                                                            test_size = 0.3, random_state = 1234567 + x)
    
    # Realizar la selección stepwise utilizando el criterio BIC en la submuestra.
    modelo = lm_backward(y_train2.astype(int), x_train2, var_cont, var_categ, interacciones_unicas, 'BIC')
    
    # Almacenar las variables seleccionadas y la fórmula correspondiente.
    variables_seleccionadas['Variables'].append(modelo['Variables'])
    variables_seleccionadas['Formula'].append(sorted(modelo['Modelo'].model.exog_names))

# Unir las variables en las fórmulas seleccionadas en una sola cadena.
variables_seleccionadas['Formula'] = list(map(lambda x: '+'.join(x), variables_seleccionadas['Formula']))
    
# Calcular la frecuencia de cada fórmula y ordenarlas por frecuencia.
frecuencias = Counter(variables_seleccionadas['Formula'])
frec_ordenada = pd.DataFrame(list(frecuencias.items()), columns = ['Formula', 'Frecuencia'])
frec_ordenada = frec_ordenada.sort_values('Frecuencia', ascending = False).reset_index()

# Identificar las dos modelos más frecuentes y las variables correspondientes.
var_1 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][0])]
var_2 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][1])]

# Si quisiéramos mostrar los tres modelos más frecuentes añadiríamos la siguiente línea de código
# var_3 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
#    frec_ordenada['Formula'][2])]

# ============================================================================
# De las 30 repeticiones, las 2 que más se repiten son:
#   1)  Clasificacion', 'CalifProductor2', ('Etiqueta', 'Clasificacion'), ('Densidad', 'Etiqueta')
#   2)  'CalifProductor2', ('Densidad', 'Clasificacion'), ('Etiqueta', 'Clasificacion'), ('Densidad', 'Etiqueta'), ('Acidez', 'pH')


## Comparacion final, tomo el ganador de antes y los nuevos candidatos
results = pd.DataFrame({
    'Rsquared': []
    , 'Resample': []
    , 'Modelo': []
})
for rep in range(20):
    modelo1 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloBackBIC['Variables']['cont']
        , modeloBackBIC['Variables']['categ']
        , modeloBackBIC['Variables']['inter']
    )
    modelo2 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , var_1['cont']
        , var_1['categ']
        , var_1['inter']
    )
    modelo3 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , var_2['cont']
        , var_2['categ']
        , var_2['inter']
    )
    
    results_rep = pd.DataFrame({
        'Rsquared': modelo1 + modelo2 + modelo3 
        , 'Resample': ['Rep' + str((rep + 1))]*5*3
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 
    })
    results = pd.concat([results, results_rep], axis = 0)
     

# Boxplot de la validacion cruzada 
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráfico
# Agrupa los valores de Rsquared por modelo
grupo_metrica = results.groupby('Modelo')['Rsquared']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('Rsquared')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico  

# Observamos esto numéricamente así como el número de parámetros de cada modelo para elegir el ganador

# Calcular la media de las métricas R-squared por modelo
media_r2_v2 = results.groupby('Modelo')['Rsquared'].mean()
print (media_r2_v2)
# Calcular la desviación estándar de las métricas R-squared por modelo
std_r2_v2 = results.groupby('Modelo')['Rsquared'].std()
print(std_r2_v2)
# Contar el número de parámetros en cada modelo
num_params_v2 = [len(modeloStepBIC_trans['Modelo'].params), 
                 len(frec_ordenada['Formula'][0].split('+')),
                 len(frec_ordenada['Formula'][1].split('+'))]

print(num_params_v2)

# Una vez decidido el mejor modelo, hay que evaluarlo 
ModeloGanador = modeloBackBIC

# Vemos los coeficientes del modelo ganador
ModeloGanador['Modelo'].summary()
# Todos los parámetros del modelo son significativos

# Evaluamos la estabilidad del modelo a partir de las diferencias en train y test
Rsq(ModeloGanador['Modelo'], y_train, ModeloGanador['X'])

x_test_modeloganador = crear_data_modelo(x_test, ModeloGanador['Variables']['cont'], 
                                                ModeloGanador['Variables']['categ'], 
                                                ModeloGanador['Variables']['inter'])
Rsq(ModeloGanador['Modelo'], y_test, x_test_modeloganador)   






### PARTE 2: REGRESIÓN LINEAL ###
    
    
# Cargo las librerias 
from sklearn.model_selection import train_test_split


# Cargo las funciones que voy a utilizar despues
from FuncionesMineria import (graficoVcramer, mosaico_targetbinaria, boxplot_targetbinaria, 
                           hist_targetbinaria, Transf_Auto, lm, Rsq, validacion_cruzada_lm,
                           modelEffectSizes, crear_data_modelo, Vcramer)
    

# Parto de los datos ya depurados del apartado anterior almacenados en el pickle
with open('DatosEleccionesEspana.pickle', 'rb') as f:
    datosRL = pickle.load(f)

# Defino las variables objetivo QUE QUIERO ESTUDIAR y las elimino del conjunto de datos input
'''
Las variables que he elegido son:
    AbstentionPtge: porque quiero estudiar que afecta a que haya mayor o menor abstención en un proceso electoral
    Derecha: quiero saber que factores actúan a la hora de elegir un partido de derechas
'''
varObjCont = datosRL['AbstentionPtge']
varObjBin = datosRL['Derecha']
datos_inputRL = datosRL.drop(['AbstentionPtge', 'Derecha'], axis = 1) 
 
# Genera una lista con los nombres de las variables.
variablesRL = list(datos_inputRL.columns)  

# Obtengo la importancia de las variables
graficoVcramer(datos_inputRL, varObjBin) #Valor de la V de Cramer para cada variable
'''
En este caso, para la varible objetivo binaria Derecha, la única varible que presenta un valor del estadístico
de Cramer interesante es CCAA, siendo de carácter categórico
'''
graficoVcramer(datos_inputRL, varObjCont) #Valor de la V de Cramer para cada variable
'''
En este caso, para la varible objetivo continua AbstentionPtge, nos encontramos con valores bajos, siendo los más
destacados en las variables:
    ComercTTEHosteleria
    Servicios
    Construccion
Siendo todas ellas de carácter cualitativo
'''
#V de Cramer mide las relaciones lineales y no lineales de cada variable con la variable objetivo

# Crear un DataFrame para almacenar los resultados del coeficiente V de Cramer
VCramer = pd.DataFrame(columns=['Variable', 'Objetivo', 'Vcramer'])

for variable in variablesRL:
    v_cramer = Vcramer(datos_inputRL[variable], varObjCont)
    VCramer = VCramer.append({'Variable': variable, 'Objetivo': varObjCont.name, 'Vcramer': v_cramer},
                             ignore_index=True)
    
for variable in variablesRL:
    v_cramer = Vcramer(datos_inputRL[variable], varObjBin)
    VCramer = VCramer._append({'Variable': variable, 'Objetivo': varObjBin.name, 'Vcramer': v_cramer}, 
                             ignore_index=True)
    
# Veo graficamente el efecto de dos variables cualitativas sobre la binaria
# Tomo las variables con más y menos relación con la variable objetivo Binaria
mosaico_targetbinaria(datos_inputRL['CCAA'], varObjBin, 'CCAA')
#Barras muy diferenciadas, podemos concluir que si que influye en la variable objetivo

# Veo graficamente el efecto de dos variables cuantitativas sobre la binaria
boxplot_targetbinaria(datos_inputRL['Age_over65_pct'], varObjBin, 'Age_over65_pct')
#Gráficos muy diferenciados, hay influencia clara
boxplot_targetbinaria(datos_inputRL['Servicios'], varObjBin, 'Servicios')
#Gráficos muy diferenciados (un poco menos), hay influencia clara

hist_targetbinaria(datos_inputRL['Age_over65_pct'], varObjBin, 'Age_over65_pct')
#Comportamiento muy diferenciados
hist_targetbinaria(datos_inputRL['Servicios'], varObjBin, 'Servicios')
#Comportamiento muy diferenciados

# Correlación entre todas las variables numéricas frente a la objetivo continua.
# Obtener las columnas numéricas del DataFrame 'datos_inputRL'
numericas = datos_inputRL.select_dtypes(include=['int', 'float']).columns
# Calcular la matriz de correlación de Pearson entre la variable objetivo continua ('varObjCont') y las variables numéricas
matriz_corr = pd.concat([varObjCont, datos_inputRL[numericas]], axis = 1).corr(method = 'pearson')
# Crear una máscara para ocultar la mitad superior de la matriz de correlación (triangular superior)
mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
# Crear una figura para el gráfico con un tamaño de 8x6 pulgadas
plt.figure(figsize=(8, 6))
# Establecer el tamaño de fuente en el gráfico
sns.set(font_scale=1.2)
# Crear un mapa de calor (heatmap) de la matriz de correlación
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f", cbar=True, mask=mask)
# Establecer el título del gráfico
plt.title("Matriz de correlación")
# Mostrar el gráfico de la matriz de correlación
plt.show()
#Existe gran variedad de varibles que tienen una elevada correlación


#ELIMINO LAS VARIABLES NO INFLUYENTES
datos_inputRL2 = datos_inputRL['Age_under19_Ptge', 'Age_over65_pct', 'CCAA', 'ActividadPpal', 
                               'Construccion', 'ComercTTEHosteleria','totalEmpresas', 'Industria',
                               'Servicios','Pob2010']

# Lista de columnas que deseas mantener en el nuevo DataFrame
columnas_deseadas = ['Age_under19_Ptge', 'Age_over65_pct', 'CCAA', 'ActividadPpal',
                     'Construccion', 'ComercTTEHosteleria', 'totalEmpresas', 'Industria',
                     'Servicios', 'Pob2010']

# Crear un nuevo DataFrame (datos_inputRL2) con las columnas deseadas
datos_inputRL2 = datos_inputRL.loc[:, columnas_deseadas].copy()

numericas = ['Age_under19_Ptge', 'Age_over65_pct', 
                               'Construccion', 'ComercTTEHosteleria','totalEmpresas', 'Industria',
                               'Servicios','Pob2010']



# Busco las mejores transformaciones para las variables numericas con respesto a los dos tipos de variables
input_cont = pd.concat([datos_inputRL2, Transf_Auto(datos_inputRL2[numericas], varObjCont)], axis = 1)
input_bin = pd.concat([datos_inputRL2, Transf_Auto(datos_inputRL2[numericas], varObjBin)], axis = 1)


# Creamos conjuntos de datos que contengan las variables explicativas y una de las variables objetivo y los guardamos
todo_contE = pd.concat([input_cont, varObjCont], axis = 1)
'''
todo_bin = pd.concat([input_bin, varObjBin], axis = 1)
with open('todo_bin.pickle', 'wb') as archivo:
    pickle.dump(todo_bin, archivo)
'''
with open('todo_contE.pickle', 'wb') as archivo:
    pickle.dump(todo_contE, archivo)
    

## Comenzamos con la regresion lineal

# Obtengo la particion
x_train, x_test, y_train, y_test = train_test_split(datos_inputRL, np.ravel(varObjCont), test_size = 0.2, random_state = 123456)

# Construyo un modelo preliminar con todas las variables (originales)
# Indico la tipología de las variables (numéricas o categóricas)
var_cont1 = ['CodigoProvincia', 'Population', 'TotalCensus', 'Age_0-4_Ptge', 'Age_under19_Ptge',
       'Age_19_65_pct', 'Age_over65_pct', 'WomanPopulationPtge',
       'ForeignersPtge', 'SameComAutonPtge', 'SameComAutonDiffProvPtge',
       'DifComAutonPtge', 'UnemployLess25_Ptge', 'Unemploy25_40_Ptge',
       'UnemployMore40_Ptge', 'AgricultureUnemploymentPtge',
       'IndustryUnemploymentPtge', 'ConstructionUnemploymentPtge',
       'ServicesUnemploymentPtge', 'totalEmpresas', 'Industria',
       'Construccion', 'ComercTTEHosteleria', 'Servicios', 'inmuebles',
       'Pob2010', 'SUPERFICIE', 'PobChange_pct', 'PersonasInmueble',
       'Explotaciones']
var_categ1 = ['CCAA', 'ActividadPpal', 'Densidad']

# Creo el modelo
modelo1 = lm(y_train, x_train, var_cont1, var_categ1)
# Visualizamos los resultado del modelo
modelo1['Modelo'].summary()


# Calculamos la medida de ajuste R^2 para los datos de entrenamiento
Rsq(modelo1['Modelo'], y_train, modelo1['X'])
#Valor de 0.356, bastante bajo

# Preparamos los datos test para usar en el modelo
x_test_modelo1 = crear_data_modelo(x_test, var_cont1, var_categ1)
# Calculamos la medida de ajuste R^2 para los datos test
Rsq(modelo1['Modelo'], y_test, x_test_modelo1)
#Se obtiene un valor 0.4774

'''
R2 -------------> 0.352
R2 training ----> 0.356
P --------------> 96.01, 0.00
Durbin-Watson --> 1.966
Jarque-Bera ----> 517.153, 5.03e-113

'''

# Nos fijamos en la importancia de las variables
modelEffectSizes(modelo1, y_train, x_train, var_cont1, var_categ1) #Determina la importancia de cada una de las variables indicando cuanto disminuiria el valor de r2 si eliminamos cada una de las variables explicativas presentes en el modelo construido


# Vamos a probar un modelo con menos variables. Recuerdo el grafico de Cramer
graficoVcramer(datos_input, varObjCont) # Pruebo con las mas importantes

# Construyo el segundo modelo
var_cont2 = ['TotalCensus', 'Age_under19_Ptge','Age_19_65_pct', 'Age_over65_pct', 
             'WomanPopulationPtge', 'ForeignersPtge', 'SameComAutonPtge', 
             'SameComAutonDiffProvPtge',
       'DifComAutonPtge', 'UnemployLess25_Ptge', 
       'IndustryUnemploymentPtge', 'ConstructionUnemploymentPtge',
       'ServicesUnemploymentPtge', 'totalEmpresas', 'Industria',
       'Servicios', 'inmuebles',
       'SUPERFICIE', 'Explotaciones']
var_categ2 = ['CCAA', 'ActividadPpal']
modelo2 = lm(y_train, x_train, var_cont2, var_categ2)
modelEffectSizes(modelo2, y_train, x_train, var_cont2, var_categ2)
modelo2['Modelo'].summary()
Rsq(modelo2['Modelo'], y_train, modelo2['X'])
x_test_modelo2 = crear_data_modelo(x_test, var_cont2, var_categ2)
Rsq(modelo2['Modelo'], y_test, x_test_modelo2)

'''
R2 -------------> 0.350
R2 training ----> 0.354
P --------------> 116.2, 0.00
Durbin-Watson --> 1.962
Jarque-Bera ----> 537.665, 1.77e-117

'''

# Pruebo un modelo con menos variables, basandome en la importancia de las variables
var_cont3 = []
var_categ3 = ['CCAA']
modelo3 = lm(y_train, x_train, var_cont3, var_categ3)
modelo3['Modelo'].summary()
Rsq(modelo3['Modelo'], y_train, modelo3['X'])
x_test_modelo3 = crear_data_modelo(x_test, var_cont3, var_categ3)
Rsq(modelo3['Modelo'], y_test, x_test_modelo3)
'''
R2 -------------> 0.268
R2 training ----> 0.272
P --------------> 263.6, 0.00
Durbin-Watson --> 1.942
Jarque-Bera ----> 351.669, 4.32e-77

'''


# Pruebo con una interaccion sobre el anterior
# Se podrian probar todas las interacciones dos a dos
var_cont4 = ['SameComAutonPtge', 'DifComAutonPtge', 'ConstructionUnemploymentPtge',
       'SUPERFICIE', 'Explotaciones']
var_categ4 = ['CCAA', 'ActividadPpal']
var_interac4 = [('CCAA', 'SUPERFICIE')] 
modelo4 = lm(y_train, x_train, var_cont4, var_categ4, var_interac4)
modelo4['Modelo'].summary()
Rsq(modelo4['Modelo'], y_train, modelo4['X'])
x_test_modelo4 = crear_data_modelo(x_test, var_cont4, var_categ4, var_interac4)
Rsq(modelo4['Modelo'], y_test, x_test_modelo4)
'''
R2 -------------> 0.340
R2 training ----> 0.346
P --------------> 133.2, 0.00
Durbin-Watson --> 1.957
Jarque-Bera ----> 553.015, 8.21e-121

'''

var_cont5 = ['SameComAutonPtge', 'DifComAutonPtge', 'ConstructionUnemploymentPtge',
       'SUPERFICIE', 'Explotaciones']
var_categ5 = ['CCAA', 'ActividadPpal'] 
modelo5 = lm(y_train, x_train, var_cont5, var_categ5)
modelo5['Modelo'].summary()
Rsq(modelo5['Modelo'], y_train, modelo5['X'])
x_test_modelo5 = crear_data_modelo(x_test, var_cont5, var_categ5)
Rsq(modelo5['Modelo'], y_test, x_test_modelo5)
'''
R2 -------------> 0.333
R2 training ----> 0.344
P --------------> 202.4, 0.00
Durbin-Watson --> 1.955
Jarque-Bera ----> 534.662, 7.94e-117

'''


# VALIDACIÓN CRUZADA repetida para ver que modelo es mejor
# Crea un DataFrame vacío para almacenar resultados
results = pd.DataFrame({
    'Rsquared': [],
    'Resample': [],
    'Modelo': []
})

# Realiza el siguiente proceso 20 veces (representado por el bucle `for rep in range(20)`)
for rep in range(20):
    # Realiza validación cruzada en cuatro modelos diferentes y almacena sus R-squared en listas separadas
    modelo1VC = validacion_cruzada_lm(5, x_train, y_train, var_cont1, var_categ1)
    modelo2VC = validacion_cruzada_lm(5, x_train, y_train, var_cont2, var_categ2)
    modelo3VC = validacion_cruzada_lm(5, x_train, y_train, var_cont3, var_categ3)
    modelo4VC = validacion_cruzada_lm(5, x_train, y_train, var_cont4, var_categ4, var_interac4)
    modelo5VC = validacion_cruzada_lm(5, x_train, y_train, var_cont5, var_categ5)

    
    # Crea un DataFrame con los resultados de validación cruzada para esta repetición
    results_rep = pd.DataFrame({
        'Rsquared': modelo1VC + modelo2VC + modelo3VC + modelo4VC + modelo5VC,
        'Resample': ['Rep' + str((rep + 1))] * 5 * 5,  # Etiqueta de repetición
        'Modelo': [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5 + [5] * 5  # Etiqueta de modelo (1, 2, 3 o 4)
    })
    
    # Concatena los resultados de esta repetición al DataFrame principal 'results'
    results = pd.concat([results, results_rep], axis=0)
#Se obtenie un R2 en cada modelo para cada una de las repeticiones (20 en total) en las 5 submuestras

    
# Boxplot de la validación cruzada
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráfico
# Agrupa los valores de R-squared por modelo
grupo_metrica = results.groupby('Modelo')['Rsquared']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('Rsquared')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico 
    

# Calcular la media de las métricas R-squared por modelo
media_r2 = results.groupby('Modelo')['Rsquared'].mean()
print(media_r2)
# Calcular la desviación estándar de las métricas R-squared por modelo
std_r2 = results.groupby('Modelo')['Rsquared'].std()
print(std_r2)

# Contar el número de parámetros en cada modelo
num_params = [len(modelo1['Modelo'].params), len(modelo2['Modelo'].params), 
             len(modelo3['Modelo'].params), len(modelo4['Modelo'].params), 
             len(modelo5['Modelo'].params)]


# Vemos los coeficientes del modelo ganador
modelo5['Modelo'].summary()

# Evaluamos la estabilidad del modelo a partir de las diferencias en train y test:
Rsq(modelo5['Modelo'], y_train, modelo5['X'])
Rsq(modelo5['Modelo'], y_test, x_test_modelo5)

# Vemos las variables mas importantes del modelo ganador
modelEffectSizes(modelo5, y_train, x_train, var_cont5, var_categ5)

    







### PARTE 3: REGRESIÓN LOGÍSTICA ###

# Cargo las funciones que voy a utilizar despues
from FuncionesMineria import (graficoVcramer, impVariablesLog, pseudoR2, glm, summary_glm, 
                           validacion_cruzada_glm, sensEspCorte, crear_data_modelo, curva_roc)

# Cargo los datos depurados (incluidas las mejores transformaciones de las 
# variables numericas respecto a la binaria)
with open('DatosEleccionesEspana.pickle', 'rb') as f:
    todo = pickle.load(f)

# Identifico la variable objetivo y la elimino de mi conjunto de datos.
varObjBin = todo['Derecha']
todo = todo.drop(['Derecha', 'AbstentionPtge'], axis = 1)


# Veo el reparto original. Compruebo que la variable objetivo tome valor 1 para el evento y 0 para el no evento
pd.DataFrame({
    'n': varObjBin.value_counts()
    , '%': varObjBin.value_counts(normalize = True)
})
#Se obtiene un 62% de SI más votos a la derecha y un 38% de NO

# Obtengo la particion
x_train, x_test, y_train, y_test = train_test_split(todo, varObjBin, test_size = 0.2, random_state = 1234567)
# Indico que la variable respuesta es numérica 
y_train, y_test = y_train.astype(int), y_test.astype(int)

# Construyo un modelo preliminar con todas las variables (originales)
# Indico la tipología de las variables (numéricas o categóricas)
var_cont1 = ['CodigoProvincia', 'Population', 'TotalCensus', 'Age_0-4_Ptge', 'Age_under19_Ptge',
       'Age_19_65_pct', 'Age_over65_pct', 'WomanPopulationPtge',
       'ForeignersPtge', 'SameComAutonPtge', 'SameComAutonDiffProvPtge',
       'DifComAutonPtge', 'UnemployLess25_Ptge', 'Unemploy25_40_Ptge',
       'UnemployMore40_Ptge', 'AgricultureUnemploymentPtge',
       'IndustryUnemploymentPtge', 'ConstructionUnemploymentPtge',
       'ServicesUnemploymentPtge', 'totalEmpresas', 'Industria',
       'Construccion', 'ComercTTEHosteleria', 'Servicios', 'inmuebles',
       'Pob2010', 'SUPERFICIE', 'PobChange_pct', 'PersonasInmueble',
       'Explotaciones']
var_categ1 = ['CCAA', 'ActividadPpal', 'Densidad']

# Creo el modelo inicial
modeloInicial = glm(y_train, x_train, var_cont1, var_categ1)
# Visualizamos los resultado del modelo
summary_glm(modeloInicial['Modelo'], y_train, modeloInicial['X'])
#Se puede observar el p valor de cada variable: *** significación al 99.9%, ** 99%, * 95% y . 90%

# Calculamos la medida de ajuste R^2 para los datos de entrenamiento para comprobar la bondad de ajuste
pseudoR2(modeloInicial['Modelo'], modeloInicial['X'], y_train)

# Preparamos los datos test para usar en el modelo (construir variabes dummy para las variables categóricas)
x_test_modeloInicial = crear_data_modelo(x_test, var_cont1, var_categ1)

# Calculamos la medida de ajuste R^2 para los datos test
pseudoR2(modeloInicial['Modelo'], x_test_modeloInicial, y_test)

# Calculamos el número de parámetros utilizados en el modelo.
len(modeloInicial['Modelo'].coef_[0])


'''
R2 -------------> 0.431
R2 training ----> 0.397
num parametros -> 43
'''

# Fijandome en la significacion de las variables, el modelo con las variables mas significativas queda
var_cont2 = ['Population', 'TotalCensus', 
       'Age_19_65_pct', 'WomanPopulationPtge',
       'ForeignersPtge', 
       'DifComAutonPtge', 'UnemployLess25_Ptge', 'Unemploy25_40_Ptge',
       'UnemployMore40_Ptge', 'AgricultureUnemploymentPtge',
       'totalEmpresas', 'Industria',
       'Servicios', 'inmuebles',
       'Pob2010', 'PobChange_pct', 
       'Explotaciones']
var_categ2 = ['CCAA', 'ActividadPpal']

modelo2 = glm(y_train, x_train, var_cont2, var_categ2)

summary_glm(modelo2['Modelo'], y_train, modelo2['X'])


pseudoR2(modelo2['Modelo'], modelo2['X'], y_train)

x_test_modelo2 = crear_data_modelo(x_test, var_cont2, var_categ2)
pseudoR2(modelo2['Modelo'], x_test_modelo2, y_test)

len(modelo2['Modelo'].coef_[0])


'''
R2 -------------> 0.429
R2 training ----> 0.399
num parametros -> 28
AROC -----------> 0.884
'''

# Calculamos y representamos la importancia de las variables en el modelo
impVariablesLog(modelo2, y_train, x_train, var_cont2, var_categ2)

# Calculamos el area bajo la curva ROC y representamos
AUC2 = curva_roc(x_test_modelo2, y_test, modelo2)

# Miro el grafico V de Cramer para ver las variables mas importantes
graficoVcramer(todo, varObjBin) 

#Las variables más relevantes obtenidas en la función anterior son:
var_cont3 = ['Age_under19_Ptge','Age_over65_pct', 'ForeignersPtge']
var_categ3 = ['CCAA', 'ActividadPpal']

modelo3 = glm(y_train, x_train, var_cont3, var_categ3)

summary_glm(modelo3['Modelo'], y_train, modelo3['X'])

pseudoR2(modelo3['Modelo'], modelo3['X'], y_train)

x_test_modelo3 = crear_data_modelo(x_test, var_cont3, var_categ3)
pseudoR2(modelo3['Modelo'], x_test_modelo3, y_test)

len(modelo3['Modelo'].coef_[0])

# Calculamos el area bajo la curva ROC y representamos
AUC3 = curva_roc(x_test_modelo3, y_test, modelo3)

'''
R2 -------------> 0.419
R2 training ----> 0.392
num parametros -> 14
AROC -----------> 0.879
'''


# Pruebo alguna interaccion sobre el modelo 3
var_cont4 = var_cont3
var_categ4 = var_categ3
var_interac4 = [('Age_over65_pct', 'CCAA')]
modelo4 = glm(y_train, x_train, var_cont4, var_categ4, var_interac4)

summary_glm(modelo4['Modelo'], y_train, modelo4['X'])
pseudoR2(modelo4['Modelo'], modelo4['X'], y_train)

x_test_modelo4 = crear_data_modelo(x_test, var_cont4, var_categ4, var_interac4)
pseudoR2(modelo4['Modelo'], x_test_modelo4, y_test)

len(modelo4['Modelo'].coef_[0])

# Calculamos el area bajo la curva ROC y representamos
AUC4 = curva_roc(x_test_modelo4, y_test, modelo4)

'''
R2 -------------> 0.428
R2 training ----> 0.403
num parametros -> 23
AROC -----------> 0.883
'''


# Pruebo uno con las variables mas importantes del 2 
var_cont5 = []
var_categ5 = ['CCAA']
modelo5 = glm(y_train, x_train, var_cont5, var_categ5)
summary_glm(modelo5['Modelo'], y_train, modelo5['X'])
pseudoR2(modelo5['Modelo'], modelo5['X'], y_train)

x_test_modelo5 = crear_data_modelo(x_test, var_cont5, var_categ5)
pseudoR2(modelo5['Modelo'], x_test_modelo5, y_test)

len(modelo5['Modelo'].coef_[0])

# Calculamos el area bajo la curva ROC y representamos
AUC5 = curva_roc(x_test_modelo5, y_test, modelo5)
'''
R2 -------------> 0.407
R2 training ----> 0.377
num parametros -> 9
AROC -----------> 0.866
'''



# Mejor modelo según el Área bajo la Curva ROC
AUC1 = curva_roc(x_test_modeloInicial, y_test, modeloInicial)
AUC2 = curva_roc(x_test_modelo2, y_test, modelo2)
AUC3 = curva_roc(x_test_modelo3, y_test, modelo3)
AUC4 = curva_roc(x_test_modelo4, y_test, modelo4)
AUC5 = curva_roc(x_test_modelo5, y_test, modelo5)
#El área es muy similar para todos los modelos, no se puede determinar cual es el mejor

# Validacion cruzada 
# Crea un DataFrame vacío para almacenar resultados
results = pd.DataFrame({
    'AUC': []
    , 'Resample': []
    , 'Modelo': []
})

# Realiza el siguiente proceso 20 veces
for rep in range(20):
    # Realiza validación cruzada en cinco modelos diferentes y almacena sus R-squared en listas separadas
    modelo1VC = validacion_cruzada_glm(5, x_train, y_train, var_cont1, var_categ1)
    modelo2VC = validacion_cruzada_glm(5, x_train, y_train, var_cont2, var_categ2)
    modelo3VC = validacion_cruzada_glm(5, x_train, y_train, var_cont3, var_categ3)
    modelo4VC = validacion_cruzada_glm(5, x_train, y_train, var_cont4, var_categ4, var_interac4)
    modelo5VC = validacion_cruzada_glm(5, x_train, y_train, var_cont5, var_categ5)
    
    # Crea un DataFrame con los resultados de validación cruzada para esta repetición
    results_rep = pd.DataFrame({
        'AUC': modelo1VC + modelo2VC + modelo3VC + modelo4VC + modelo5VC 
        , 'Resample': ['Rep' + str((rep + 1))]*5*5  # Etiqueta de repetición (5 repeticiones 6 modelos)
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 + [4]*5 + [5]*5 # Etiqueta de modelo (6 modelos 5 repeticiones)
    })
    results = pd.concat([results, results_rep], axis = 0)
    

# Boxplot de la validacion cruzada 
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráficoç
# Agrupa los valores de AUC por modelo
grupo_metrica = results.groupby('Modelo')['AUC']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('AUC')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico  
 
    
# Calcular la media del AUC por modelo
results.groupby('Modelo')['AUC'].mean()
# Calcular la desviación estándar del AUC por modelo
results.groupby('Modelo')['AUC'].std()    
# Contar el número de parámetros en cada modelo
num_params = [len(modeloInicial['Modelo'].coef_[0]), len(modelo2['Modelo'].coef_[0]), len(modelo3['Modelo'].coef_[0]), 
 len(modelo4['Modelo'].coef_[0]), len(modelo5['Modelo'].coef_[0])]

print(num_params)

## Buscamos el mejor punto de corte

# Probamos dos
sensEspCorte(modelo3['Modelo'], x_test, y_test, 0.4, var_cont3, var_categ3)
sensEspCorte(modelo3['Modelo'], x_test, y_test, 0.6, var_cont3, var_categ3)

# Generamos una rejilla de puntos de corte
posiblesCortes = np.arange(0, 1.01, 0.01).tolist()  # Generamos puntos de corte de 0 a 1 con intervalo de 0.01
rejilla = pd.DataFrame({
    'PtoCorte': [],
    'Accuracy': [],
    'Sensitivity': [],
    'Specificity': [],
    'PosPredValue': [],
    'NegPredValue': []
})  # Creamos un DataFrame para almacenar las métricas para cada punto de corte

for pto_corte in posiblesCortes:  # Iteramos sobre los puntos de corte
    rejilla = pd.concat(
        [rejilla, sensEspCorte(modelo5['Modelo'], x_test, y_test, pto_corte, var_cont5, var_categ5)],
        axis=0
    )  # Calculamos las métricas para el punto de corte actual y lo agregamos al DataFrame

rejilla['Youden'] = rejilla['Sensitivity'] + rejilla['Specificity'] - 1  # Calculamos el índice de Youden
rejilla.index = list(range(len(rejilla)))  # Reindexamos el DataFrame para que los índices sean consecutivos



plt.plot(rejilla['PtoCorte'], rejilla['Youden'])
plt.xlabel('Posibles Cortes')
plt.ylabel('Youden')
plt.title('Youden')
plt.show()

plt.plot(rejilla['PtoCorte'], rejilla['Accuracy'])
plt.xlabel('Posibles Cortes')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.show()

plt.plot(rejilla['PtoCorte'], rejilla['Sensitivity'])
plt.xlabel('Posibles Cortes')
plt.ylabel('Sensitivity')
plt.title('Sensitivity')
plt.show()

plt.plot(rejilla['PtoCorte'], rejilla['Specificity'])
plt.xlabel('Posibles Cortes')
plt.ylabel('Specificity')
plt.title('Specificity')
plt.show()

rejilla['PtoCorte'][rejilla['Youden'].idxmax()]
rejilla['PtoCorte'][rejilla['Accuracy'].idxmax()]
rejilla['PtoCorte'][rejilla['Specificity'].idxmax()]

# Los comparamos
sensEspCorte(modelo3['Modelo'], x_test, y_test, 0.46, var_cont3, var_categ3)
sensEspCorte(modelo3['Modelo'], x_test, y_test, 0.46, var_cont3, var_categ3)

# Vemos las variables mas importantes del modelo ganador
impVariablesLog(modelo3, y_train, x_train, var_cont3, var_categ3)


# Vemos los coeficientes del modelo ganador
coeficientes = modelo3['Modelo'].coef_
nombres_caracteristicas = crear_data_modelo(x_train, var_cont3, var_categ3).columns  # Suponiendo que X_train es un DataFrame de pandas
# Imprime los nombres de las características junto con sus coeficientes
for nombre, coef in zip(nombres_caracteristicas, coeficientes[0]):
    print(f"Variable: {nombre}, Coeficiente: {coef}")

# Evaluamos la estabilidad del modelo a partir de las diferencias en train y test:
pseudoR2(modelo3['Modelo'], modelo3['X'], y_train)
pseudoR2(modelo3['Modelo'], x_test_modelo3, y_test)

# Calculamos la diferencia del Area bajo la curva ROC en train y test
curva_roc(crear_data_modelo(x_train, var_cont3, var_categ3), y_train, modelo3)
curva_roc(x_test_modelo3, y_test, modelo3)

# Calculamos la diferencia de las medidas de calidad entre train y test 
sensEspCorte(modelo3['Modelo'], x_train, y_train, 0.5, var_cont3, var_categ3)
sensEspCorte(modelo3['Modelo'], x_test, y_test, 0.5, var_cont3, var_categ3)


    



















### SELECCIÓN DE VARIABLES
# Cargo las librerias 
import os
import pickle
from sklearn.model_selection import train_test_split
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


# Cargo las funciones que voy a utilizar
from FuncionesMineria import (Rsq, lm, lm_forward, lm_backward, lm_stepwise, validacion_cruzada_lm,
                           crear_data_modelo)

# Cargo los datos depurados
with open('todo_contE.pickle', 'rb') as f:
    todo = pickle.load(f)

# Identifico la variable objetivo y la elimino del conjunto de datos
varObjCont = datosRL['AbstentionPtge']
#todo = todo.drop('Beneficio', axis = 1)

# Identifico las variables continuas
var_cont = ['Age_under19_Ptge', 'Age_over65_pct', 
                               'Construccion', 'ComercTTEHosteleria','totalEmpresas', 'Industria',
                               'Servicios','Pob2010', 'xAge_under19_Ptge', 'sqrtxAge_over65_pct', 
                                'logxConstruccion', 'logxComercTTEHosteleria','logxtotalEmpresas', 'logxIndustria',
                                 'logxServicios','logxPob2010']



# Identifico las variables continuas sin transformar
var_cont_sin_transf = ['Age_under19_Ptge', 'Age_over65_pct', 
                               'Construccion', 'ComercTTEHosteleria','totalEmpresas', 'Industria',
                               'Servicios','Pob2010']

# Identifico las variables categóricas
var_categ = ['CCAA', 'ActividadPpal']

# Hago la particion
x_train, x_test, y_train, y_test = train_test_split(todo, varObjCont, test_size = 0.2, random_state = 1234567)



# Seleccion de variables Stepwise, métrica AIC
modeloStepAIC = lm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ, [], 'AIC')
# Resumen del modelo
modeloStepAIC['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepAIC['Modelo'], y_train, modeloStepAIC['X'])

# Preparo datos test
x_test_modeloStepAIC = crear_data_modelo(x_test, modeloStepAIC['Variables']['cont'], 
                                                modeloStepAIC['Variables']['categ'], 
                                                modeloStepAIC['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepAIC['Modelo'], y_test, x_test_modeloStepAIC)

# Seleccion de variables Backward, métrica AIC
modeloBackAIC = lm_backward(y_train, x_train, var_cont_sin_transf, var_categ, [], 'AIC')
modeloBackAIC['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloBackAIC['Modelo'], y_train, modeloBackAIC['X'])


x_test_modeloBackAIC = crear_data_modelo(x_test, modeloBackAIC['Variables']['cont'], 
                                                modeloBackAIC['Variables']['categ'], 
                                                modeloBackAIC['Variables']['inter'])


Rsq(modeloBackAIC['Modelo'], y_test, x_test_modeloBackAIC)

# Comparo número de parámetros (iguales)
len(modeloStepAIC['Modelo'].params)
len(modeloBackAIC['Modelo'].params)


# Mismas variables seleccionadas, mismos parámetros, mismo modelo.
modeloStepAIC['Modelo'].params
modeloBackAIC['Modelo'].params

# Seleccion de variables Stepwise, métrica BIC
modeloStepBIC = lm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ, [], 'BIC')
# Resumen del modelo
modeloStepBIC['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepBIC['Modelo'], y_train, modeloStepBIC['X'])

# Preparo datos test
x_test_modeloStepBIC = crear_data_modelo(x_test, modeloStepBIC['Variables']['cont'], 
                                                modeloStepBIC['Variables']['categ'], 
                                                modeloStepBIC['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepBIC['Modelo'], y_test, x_test_modeloStepBIC)

# Seleccion de variables Backward, métrica BIC
modeloBackBIC = lm_backward(y_train, x_train, var_cont_sin_transf, var_categ, [], 'BIC')
# Resumen del modelo
modeloBackBIC['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloBackBIC['Modelo'], y_train, modeloBackBIC['X'])

# Preparo datos test
x_test_modeloBackBIC = crear_data_modelo(x_test, modeloBackBIC['Variables']['cont'], 
                                                modeloBackBIC['Variables']['categ'], 
                                                modeloBackBIC['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloBackBIC['Modelo'], y_test, x_test_modeloBackBIC)

# Comparo número de parámetros
len(modeloBackBIC['Modelo'].params)
len(modeloStepBIC['Modelo'].params)


modeloStepBIC['Modelo'].params
modeloBackBIC['Modelo'].params


# Comparo (R-squared)
Rsq(modeloStepAIC['Modelo'], y_train, modeloStepAIC['X'])
Rsq(modeloStepBIC['Modelo'], y_train, modeloStepBIC['X'])

Rsq(modeloStepAIC['Modelo'], y_test, x_test_modeloStepAIC)
Rsq(modeloStepBIC['Modelo'], y_test, x_test_modeloStepBIC)


# Interacciones 2 a 2 de todas las variables (excepto las continuas transformadas)
interacciones = var_cont_sin_transf + var_categ
interacciones_unicas = list(itertools.combinations(interacciones, 2)) 
  
# Seleccion de variables Stepwise, métrica AIC, con interacciones
modeloStepAIC_int = lm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ, 
                                interacciones_unicas, 'AIC')
# Resumen del modelo
modeloStepAIC_int['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepAIC_int['Modelo'], y_train, modeloStepAIC_int['X'])

# Preparo datos test
x_test_modeloStepAIC_int = crear_data_modelo(x_test, modeloStepAIC_int['Variables']['cont'], 
                                                    modeloStepAIC_int['Variables']['categ'], 
                                                    modeloStepAIC_int['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepAIC_int['Modelo'], y_test, x_test_modeloStepAIC_int)

# Seleccion de variables Stepwise, métrica BIC, con interacciones
modeloStepBIC_int = lm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ,
                                interacciones_unicas, 'BIC')
# Resumen del modelo
modeloStepBIC_int['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepBIC_int['Modelo'], y_train, modeloStepBIC_int['X'])

# Preparo datos test
x_test_modeloStepBIC_int = crear_data_modelo(x_test, modeloStepBIC_int['Variables']['cont'], 
                                                    modeloStepBIC_int['Variables']['categ'], 
                                                    modeloStepBIC_int['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepBIC_int['Modelo'], y_test, x_test_modeloStepBIC_int)
  


# Comparo los R^2 del modelo utilizando ambos criterios
Rsq(modeloStepAIC_int['Modelo'], y_train, modeloStepAIC_int['X'])
Rsq(modeloStepAIC_int['Modelo'], y_test, x_test_modeloStepAIC_int)


Rsq(modeloStepBIC_int['Modelo'], y_train, modeloStepBIC_int['X'])
Rsq(modeloStepBIC_int['Modelo'], y_test, x_test_modeloStepBIC_int)



# Comparo número de parámetros  
len(modeloStepAIC_int['Modelo'].params)
len(modeloStepBIC_int['Modelo'].params)


# Pruebo con todas las transf y las variables originales, métrica AIC
modeloStepAIC_trans = lm_stepwise(y_train, x_train, var_cont, var_categ, [], 'AIC')
# Resumen del modelo
modeloStepAIC_trans['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepAIC_trans['Modelo'], y_train, modeloStepAIC_trans['X'])

# Preparo datos test
x_test_modeloStepAIC_trans = crear_data_modelo(x_test, modeloStepAIC_trans['Variables']['cont'], 
                                                      modeloStepAIC_trans['Variables']['categ'], 
                                                      modeloStepAIC_trans['Variables']['inter'])

# R-squared del modelo para test
Rsq(modeloStepAIC_trans['Modelo'], y_test, x_test_modeloStepAIC_trans)

# Pruebo con todas las transf y las variables originales, métrica BIC
modeloStepBIC_trans = lm_stepwise(y_train, x_train, var_cont, var_categ, [], 'BIC')
# Resumen del modelo
modeloStepBIC_trans['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepBIC_trans['Modelo'], y_train, modeloStepBIC_trans['X'])

# Preparo datos test
x_test_modeloStepBIC_trans = crear_data_modelo(x_test, modeloStepBIC_trans['Variables']['cont'], 
                                                      modeloStepBIC_trans['Variables']['categ'], 
                                                      modeloStepBIC_trans['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepBIC_trans['Modelo'], y_test, x_test_modeloStepBIC_trans)


# Comparo los R^2 de los modelos utilizando ambos criterios
Rsq(modeloStepAIC_trans['Modelo'], y_train, modeloStepAIC_trans['X'])
Rsq(modeloStepAIC_trans['Modelo'], y_test, x_test_modeloStepAIC_trans)


Rsq(modeloStepBIC_trans['Modelo'], y_train, modeloStepBIC_trans['X'])
Rsq(modeloStepBIC_trans['Modelo'], y_test, x_test_modeloStepBIC_trans)


# Comparo número de parámetros  
len(modeloStepAIC_trans['Modelo'].params)
len(modeloStepBIC_trans['Modelo'].params)

# Pruebo modelo con las Transformaciones y las interacciones, métrica AIC
modeloStepAIC_transInt = lm_stepwise(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'AIC')
# Resumen del modelo
modeloStepAIC_transInt['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepAIC_transInt['Modelo'], y_train, modeloStepAIC_transInt['X'])

# Preparo datos test
x_test_modeloStepAIC_transInt = crear_data_modelo(x_test, modeloStepAIC_transInt['Variables']['cont'], 
                                                         modeloStepAIC_transInt['Variables']['categ'], 
                                                         modeloStepAIC_transInt['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepAIC_transInt['Modelo'], y_test, x_test_modeloStepAIC_transInt)

# Pruebo modelo con las Transformaciones y las interacciones, métrica BIC
modeloStepBIC_transInt = lm_stepwise(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'BIC')
# Resumen del modelo
modeloStepBIC_transInt['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepBIC_transInt['Modelo'], y_train, modeloStepBIC_transInt['X'])


# Preparo datos test
x_test_modeloStepBIC_transInt = crear_data_modelo(x_test, modeloStepBIC_transInt['Variables']['cont'], 
                                                         modeloStepBIC_transInt['Variables']['categ'], 
                                                         modeloStepBIC_transInt['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepBIC_transInt['Modelo'], y_test, x_test_modeloStepBIC_transInt)


# Comparo los R^2 de los modelos utilizando ambos criterios
Rsq(modeloStepAIC_transInt['Modelo'], y_train, modeloStepAIC_transInt['X'])
Rsq(modeloStepAIC_transInt['Modelo'], y_test, x_test_modeloStepAIC_transInt)


Rsq(modeloStepBIC_transInt['Modelo'], y_train, modeloStepBIC_transInt['X'])
Rsq(modeloStepBIC_transInt['Modelo'], y_test, x_test_modeloStepBIC_transInt)


# Comparo número de parámetros  
len(modeloStepAIC_transInt['Modelo'].params)
len(modeloStepBIC_transInt['Modelo'].params)


# Hago validacion cruzada repetida para ver que modelo es mejor
# Crea un DataFrame vacío para almacenar resultados
results = pd.DataFrame({
    'Rsquared': []
    , 'Resample': []
    , 'Modelo': []
})
# Realiza el siguiente proceso 20 veces (representado por el bucle `for rep in range(20)`)

for rep in range(20):
    # Realiza validación cruzada en modelos diferentes y almacena sus R-squared en listas separadas

    modelo_stepBIC = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC['Variables']['cont']
        , modeloStepBIC['Variables']['categ']
    )
    modelo_BackBIC = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloBackBIC['Variables']['cont']
        , modeloBackBIC['Variables']['categ']
    )
    modelo_stepBIC_int = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC_int['Variables']['cont']
        , modeloStepBIC_int['Variables']['categ']
        , modeloStepBIC_int['Variables']['inter']
    )
    modelo_stepBIC_trans = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC_trans['Variables']['cont']
        , modeloStepBIC_trans['Variables']['categ']
    )
    modelo_stepBIC_transInt = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC_transInt['Variables']['cont']
        , modeloStepBIC_transInt['Variables']['categ']
        , modeloStepBIC_transInt['Variables']['inter']
    )
    # Crea un DataFrame con los resultados de validación cruzada para esta repetición

    results_rep = pd.DataFrame({
        'Rsquared': modelo_stepBIC + modelo_stepBIC_int + modelo_stepBIC_trans + modelo_stepBIC_trans + modelo_stepBIC_transInt
        , 'Resample': ['Rep' + str((rep + 1))]*5*5 # Etiqueta de repetición (5 repeticiones 6 modelos)
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 + [4]*5 + [5]*5  # Etiqueta de modelo (6 modelos 5 repeticiones)
    })
    results = pd.concat([results, results_rep], axis = 0)
    
# Boxplot de la validacion cruzada 
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráficoç
# Agrupa los valores de Rsquared por modelo
grupo_metrica = results.groupby('Modelo')['Rsquared']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('Rsquared')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico  

# Calcular la media de las métricas R-squared por modelo
media_r2 = results.groupby('Modelo')['Rsquared'].mean()
print(media_r2)
# Calcular la desviación estándar de las métricas R-squared por modelo
std_r2 = results.groupby('Modelo')['Rsquared'].std()
print(std_r2)
# Contar el número de parámetros en cada modelo
num_params = [len(modeloStepAIC['Modelo'].params), len(modeloStepBIC_int['Modelo'].params), 
 len(modeloStepAIC_trans['Modelo'].params), len(modeloStepBIC_trans['Modelo'].params), 
 len(modeloStepBIC_transInt['Modelo'].params)]

print(num_params)


## Seleccion aleatoria (se coge la submuestra de los datos de entrenamiento)
# Concretamente el 70% de los datos de entrenamiento utilizados para contruir los 
# modelos anteriores.
# El método de selección usado ha sido el Backward con el criterio BIC
# Se aplica este método a 30 submuestras diferentes

# Inicializar un diccionario para almacenar las fórmulas y variables seleccionadas.
variables_seleccionadas = {
    'Formula': [],
    'Variables': []
}

# Realizar 30 iteraciones de selección aleatoria.
for x in range(30):
    print('---------------------------- iter: ' + str(x))
    
    # Dividir los datos de entrenamiento en conjuntos de entrenamiento y prueba.
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train, 
                                                            test_size = 0.3, random_state = 1234567 + x)
    
    # Realizar la selección stepwise utilizando el criterio BIC en la submuestra.
    modelo = lm_backward(y_train2.astype(int), x_train2, var_cont, var_categ, interacciones_unicas, 'BIC')
    
    # Almacenar las variables seleccionadas y la fórmula correspondiente.
    variables_seleccionadas['Variables'].append(modelo['Variables'])
    variables_seleccionadas['Formula'].append(sorted(modelo['Modelo'].model.exog_names))

# Unir las variables en las fórmulas seleccionadas en una sola cadena.
variables_seleccionadas['Formula'] = list(map(lambda x: '+'.join(x), variables_seleccionadas['Formula']))
    
# Calcular la frecuencia de cada fórmula y ordenarlas por frecuencia.
frecuencias = Counter(variables_seleccionadas['Formula'])
frec_ordenada = pd.DataFrame(list(frecuencias.items()), columns = ['Formula', 'Frecuencia'])
frec_ordenada = frec_ordenada.sort_values('Frecuencia', ascending = False).reset_index()

# Identificar las dos modelos más frecuentes y las variables correspondientes.
var_1 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][0])]
var_2 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][1])]

# Si quisiéramos mostrar los tres modelos más frecuentes añadiríamos la siguiente línea de código
# var_3 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
#    frec_ordenada['Formula'][2])]

# ============================================================================
# De las 30 repeticiones, las 2 que más se repiten son:
#   1)  Clasificacion', 'CalifProductor2', ('Etiqueta', 'Clasificacion'), ('Densidad', 'Etiqueta')
#   2)  'CalifProductor2', ('Densidad', 'Clasificacion'), ('Etiqueta', 'Clasificacion'), ('Densidad', 'Etiqueta'), ('Acidez', 'pH')


## Comparacion final, tomo el ganador de antes y los nuevos candidatos
results = pd.DataFrame({
    'Rsquared': []
    , 'Resample': []
    , 'Modelo': []
})
for rep in range(20):
    modelo1 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloBackBIC['Variables']['cont']
        , modeloBackBIC['Variables']['categ']
        , modeloBackBIC['Variables']['inter']
    )
    modelo2 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , var_1['cont']
        , var_1['categ']
        , var_1['inter']
    )
    modelo3 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , var_2['cont']
        , var_2['categ']
        , var_2['inter']
    )
    
    results_rep = pd.DataFrame({
        'Rsquared': modelo1 + modelo2 + modelo3 
        , 'Resample': ['Rep' + str((rep + 1))]*5*3
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 
    })
    results = pd.concat([results, results_rep], axis = 0)
     

# Boxplot de la validacion cruzada 
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráfico
# Agrupa los valores de Rsquared por modelo
grupo_metrica = results.groupby('Modelo')['Rsquared']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('Rsquared')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico  

# Observamos esto numéricamente así como el número de parámetros de cada modelo para elegir el ganador

# Calcular la media de las métricas R-squared por modelo
media_r2_v2 = results.groupby('Modelo')['Rsquared'].mean()
print (media_r2_v2)
# Calcular la desviación estándar de las métricas R-squared por modelo
std_r2_v2 = results.groupby('Modelo')['Rsquared'].std()
print(std_r2_v2)
# Contar el número de parámetros en cada modelo
num_params_v2 = [len(modeloStepBIC_trans['Modelo'].params), 
                 len(frec_ordenada['Formula'][0].split('+')),
                 len(frec_ordenada['Formula'][1].split('+'))]

print(num_params_v2)

# Una vez decidido el mejor modelo, hay que evaluarlo 
ModeloGanador = modeloBackBIC

# Vemos los coeficientes del modelo ganador
ModeloGanador['Modelo'].summary()
# Todos los parámetros del modelo son significativos

# Evaluamos la estabilidad del modelo a partir de las diferencias en train y test
Rsq(ModeloGanador['Modelo'], y_train, ModeloGanador['X'])

x_test_modeloganador = crear_data_modelo(x_test, ModeloGanador['Variables']['cont'], 
                                                ModeloGanador['Variables']['categ'], 
                                                ModeloGanador['Variables']['inter'])
Rsq(ModeloGanador['Modelo'], y_test, x_test_modeloganador)    
    
    