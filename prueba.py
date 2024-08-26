import pandas as pd
import os
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

path = 'C:/Users/PC/Desktop/Master_IA/Principios_Machine_Learning/Datos_Etapa_1_csv.csv'
with open(path, 'r') as file:
    content = file.read()

content = content.replace('"', '')  # Eliminar las comillas dobles

with open(path, 'w') as file:
    file.write(content)

#Carga y verificacion del CSV
data_original = pd.read_csv(path, sep=',')
data_copy = data_original.copy()
print(data_copy.shape)
data_copy.head()
data_copy=data_copy.drop(['season','weathersit','time_of_day'], axis=1)
print(data_copy.shape)
data_copy.head()
data_copy.isna().sum()
#verificar registros duplicados
data_copy.duplicated().sum()
data_copy = data_copy.drop_duplicates()
train_lasso, test_lasso = train_test_split(data_copy, test_size=0.2, random_state=77)
x_train_lasso = train_lasso.drop(['cnt'], axis = 1)
y_train_lasso = train_lasso['cnt']
#ESTANDARIZACIÓN DE LOS DATOS
columns = x_train_lasso.columns
#creacion de objeto StandardScaler()
scaler = StandardScaler()
x_train_lasso = pd.DataFrame(scaler.fit_transform(x_train_lasso), columns = columns)
#BUSQUEDA DE HIPERPARAMETRO Y ENTRENAMIENTO
kfold = KFold(n_splits=10, shuffle= True, random_state=0)
#creacion objeto clase lasso
lasso = Lasso()
#definición de busqueda de hiperparametro
valores_alpha = [1, 2, 3, 4, 5]
param_grid_lasso = {'alpha': valores_alpha}
grid_lasso = GridSearchCV(lasso, param_grid_lasso, cv = kfold, n_jobs = 1)
grid_lasso.fit(x_train_lasso, y_train_lasso)


#mostrar mejor valor alfa
mejor_modelo_lasso = grid_lasso.best_estimator_
x_test_lasso = test_lasso.drop(['cnt'], axis = 1)
y_test_lasso = test_lasso['cnt']
x_test_lasso = pd.DataFrame(scaler.transform(x_test_lasso), columns= columns)

print("mejor parametro lasso:", grid_lasso.best_params_)
list(zip(x_train_lasso.columns, mejor_modelo_lasso.coef_))

#predicciones
y_pred_lasso = mejor_modelo_lasso.predict(x_test_lasso)
#manejor de errores
rmse_lasso = mean_squared_error(y_test_lasso, y_pred_lasso, squared = False)
mae_lasso = mean_absolute_error(y_test_lasso, y_pred_lasso)
r2_lasso = r2_score(y_test_lasso, y_pred_lasso)

print(y_pred_lasso)
print(["RMSE:", rmse_lasso], ["MAE:", mae_lasso], ["R2:", r2_lasso])