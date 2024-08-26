import pandas as pd

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, RobustScaler

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

path = '../resources/Datos_Etapa_1_csv.csv'
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
data_copy['season'] = data_copy['season'].astype('category').cat.codes
data_copy['weathersit'] = data_copy['weathersit'].astype('category').cat.codes
data_copy['time_of_day'] = data_copy['time_of_day'].astype('category').cat.codes
print(data_copy.shape)
data_copy.head()
data_copy.isna().sum()
#verificar registros duplicados
data_copy.duplicated().sum()
data_copy = data_copy.drop_duplicates()
x_pol = data_copy.drop(['cnt'], axis=1)
y_pol = data_copy['cnt']
x_train_pol, x_test_pol, y_train_pol, y_test_pol = train_test_split(x_pol, y_pol, test_size=0.2, random_state=77)
#BUSQUEDA DE HIPER PARAMETRO
polynomial_regression = make_pipeline(
    PolynomialFeatures(),
    RobustScaler(),
    LinearRegression()
)
kfold = KFold(n_splits=10, shuffle= True, random_state=0)
#espacio de busqueda
valores_busqueda = [2,3]
param_grid_pol = {'polynomialfeatures__degree': valores_busqueda}
grid_pol = GridSearchCV(polynomial_regression, param_grid_pol, cv = kfold, n_jobs = 1)
grid_pol.fit(x_train_pol, y_train_pol)
print("mejor parametro polinomial: ", grid_pol.best_params_)
mejor_modelo_pol = grid_pol.best_estimator_
#prediciones
y_pred_pol = mejor_modelo_pol.predict(x_test_pol)
#manejo de errores
rmse_pol = mean_squared_error(y_test_pol, y_pred_pol, squared = False)
mae_pol = mean_absolute_error(y_test_pol, y_pred_pol)
r2_pol = r2_score(y_test_pol, y_pred_pol)
print(y_pred_pol)
print(["RMSE:", rmse_pol], ["MAE",mae_pol], ["R2:", r2_pol])