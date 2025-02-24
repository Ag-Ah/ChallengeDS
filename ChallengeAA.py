import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from xgboost import XGBRegressor,plot_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import fastapi
from fastapi import FastAPI
import uvicorn
import pydantic
from pydantic import BaseModel


# Cargar datos
tiendas = pd.read_csv("stores data-set.csv")
caracteristicas = pd.read_csv("Features data set.csv")
ventas = pd.read_csv("sales data-set.csv")

# Convertir fechas a tipo datetime
caracteristicas['Date'] = pd.to_datetime(caracteristicas['Date'])
ventas['Date'] = pd.to_datetime(ventas['Date'])

# Unir datasets
ventas = ventas.merge(tiendas, on='Store', how='left')
ventas = ventas.merge(caracteristicas, on=['Store', 'Date'], how='left')
ventas = ventas.drop('IsHoliday_y', axis=1)
ventas = ventas.rename(columns={'IsHoliday_x': 'IsHoliday'})

# Evaluación de datos faltantes
missing_values = ventas.isnull().sum()
missing_percentage = (missing_values / len(ventas)) * 100
print("Cantidad de valores faltantes por columna:")
print(pd.DataFrame({'Total NAs': missing_values, 'Porcentaje': missing_percentage}))

# Crear nuevas características
ventas['Year'] = ventas['Date'].dt.year
ventas['Month'] = ventas['Date'].dt.month
ventas['Week'] = ventas['Date'].dt.isocalendar().week
ventas['Total_MarkDown'] = ventas[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].sum(axis=1, skipna=True)
ventas['DiscountImpact'] = ventas['Total_MarkDown'] / (ventas['Weekly_Sales'] + 1)
ventas['Size_Discount_Interaction'] = ventas['Size'] * ventas['Total_MarkDown']
ventas['Unemployment_Sales_Interaction'] = ventas['Unemployment'] * ventas['Weekly_Sales']


# Matriz de correlación entre variables
plt.figure(figsize=(12, 6))
sns.heatmap(ventas.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Matriz de Correlación entre Variables")
plt.show()

# Análisis de la variable 'Type'
grouped = ventas.groupby('Type')
average_sales = grouped.mean()
print(average_sales)
plt.figure(figsize=(8, 5))
sns.countplot(data=tiendas, x='Type',  palette='coolwarm')
plt.title("Distribución de Tipos de Tienda")
plt.xlabel("Tipo de Tienda")
plt.ylabel("Cantidad de Tiendas")
plt.show()

# Modelado de los efectos de los descuentos durante las semanas festivas
plt.figure(figsize=(10, 5))
sns.boxplot(data=ventas, x='IsHoliday', y='Total_MarkDown')
plt.title("Distribución de Descuentos en Semanas Festivas y No Festivas")
plt.xlabel("¿Es Festivo?")
plt.ylabel("Total Descuentos")
plt.show()


# Comparación de ventas en semanas festivas con y sin descuentos
ventas['Has_Discount'] = ventas['Total_MarkDown'] > 0
festive_sales = ventas[ventas['IsHoliday'] == True].groupby('Has_Discount')['Weekly_Sales'].mean()
print("Ventas promedio en semanas festivas con y sin descuentos:")
print(festive_sales)

festive_weeks = ventas[ventas['IsHoliday'] == True]
discount_ratio_festive = festive_weeks['Has_Discount'].value_counts(normalize=True) * 100
print("Proporción de semanas festivas con y sin descuentos:")
print(discount_ratio_festive)

non_festive_weeks = ventas[ventas['IsHoliday'] == False]
discount_ratio_nonfestive = non_festive_weeks['Has_Discount'].value_counts(normalize=True) * 100
print("Proporción de semanas no festivas con y sin descuentos:")
print(discount_ratio_nonfestive)


festive_sales = ventas[ventas['IsHoliday'] == True].groupby('Has_Discount')['Weekly_Sales'].mean().reset_index()
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=festive_sales, x='Has_Discount', y='Weekly_Sales', palette='coolwarm')
plt.title("Ventas Promedio en Semanas Festivas: Con y Sin Descuento")
plt.xlabel("Tiene Descuento")
plt.ylabel("Ventas Promedio Semanales")
plt.xticks(ticks=[0, 1], labels=['Sin Descuento', 'Con Descuento'])
plt.bar_label(ax.containers[0], fmt='%.2f', label_type='edge', fontsize=12, color='black')
plt.bar_label(ax.containers[1], fmt='%.2f', label_type='edge', fontsize=12, color='black')
plt.show()

#Comparación de ventas promedio en semanas no festivas con y sin descuentos
non_festive_sales = ventas[ventas['IsHoliday'] == False].groupby('Has_Discount')['Weekly_Sales'].mean().reset_index()
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=non_festive_sales, x='Has_Discount', y='Weekly_Sales', palette='coolwarm')
plt.title("Ventas Promedio en Semanas No Festivas: Con y Sin Descuento")
plt.xlabel("Tiene Descuento")
plt.ylabel("Ventas Promedio Semanales")
plt.xticks(ticks=[0, 1], labels=['Sin Descuento', 'Con Descuento'])
plt.bar_label(ax.containers[0], fmt='%.2f', label_type='edge', fontsize=12, color='black')
plt.bar_label(ax.containers[1], fmt='%.2f', label_type='edge', fontsize=12, color='black')
plt.show()

#Crear variable con lag de un año para predicción
ventas['Year_Lag'] = ventas['Year'] - 1
ventas_lag = ventas[['Store', 'Dept', 'Year_Lag', 'Week', 'Weekly_Sales']]
ventas_lag.rename(columns={'Weekly_Sales': 'Sales_Last_Year', 'Year_Lag': 'Year'}, inplace=True)
ventas = ventas.merge(ventas_lag, on=['Store', 'Dept', 'Year', 'Week'], how='left')

# Seleccionar características y eliminar columnas irrelevantes
features = ["Store", "Dept", "Size", "Temperature", "Fuel_Price", "CPI", "Unemployment", "IsHoliday", "Year", "Month", "Week", "Total_MarkDown", "Sales_Last_Year",'MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','DiscountImpact','Size_Discount_Interaction','Unemployment_Sales_Interaction']
X = ventas[features].astype(float)
y = ventas["Weekly_Sales"]

#Filtrar solo datos hasta 2011 para entrenar el modelo y prever 2012 
train_cutoff = "2011-11-30"
test_cutoff = "2012-11-30"

X_train = X[ventas['Date'] <= train_cutoff]
y_train = y[ventas['Date'] <= train_cutoff]
X_test = X[(ventas['Date'] > train_cutoff) & (ventas['Date'] <= test_cutoff)]
y_test = y[(ventas['Date'] > train_cutoff) & (ventas['Date'] <= test_cutoff)]


param_grid = {
'n_estimators': [100, 200, 300],
'learning_rate': [0.01, 0.05, 0.1],
'max_depth': [3, 6, 9],
'subsample': [0.8, 1.0],
'colsample_bytree': [0.8, 1.0]
}

xgb = XGBRegressor(random_state=42)
grid_search = GridSearchCV(xgb, param_grid, scoring='neg_mean_absolute_error', cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

#Mejor modelo encontrado
best_model = grid_search.best_estimator_
# best_model = joblib.load("modelo_ventas_xgb_optimizado.pkl")

#Evaluación del modelo
plt.figure(figsize=(14, 10))
plot_importance(best_model, max_num_features=10)
plt.title("Importancia de las Variables en XGBoost")
plt.subplots_adjust(left=0.25)
plt.yticks(fontsize=13)
plt.show()

y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mejores hiperparámetros: {best_model.get_params}")
print(f"Performance del mejor modelo:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Guardar modelo entrenado
joblib.dump(best_model, "modelo_ventas_xgb_optimizado.pkl")

# Predicción de las ventas de todas las tiendas para el año siguiente
future_dates = pd.date_range(start="2012-12-01", end="2013-12-01", freq='W')
predictions = []

for store in ventas['Store'].unique():
    for dept in ventas['Dept'].unique():
        filtered_df = ventas[(ventas['Store'] == store) & (ventas['Dept'] == dept)]
        
        if not filtered_df.empty:  
            row = filtered_df.iloc[-1].copy() 
            
            for date in future_dates:
                new_row = row.copy()  
                new_row['Year'], new_row['Month'], new_row['Week'], new_row['DayOfWeek'] = (
                    date.year, date.month, date.isocalendar().week, date.dayofweek
                )
                new_row['IsEndOfMonth'] = date.is_month_end
                new_row['Date'] = date
                predictions.append(new_row)

future_df = pd.DataFrame(predictions)
X_future = future_df[X_train.columns]
y_future_pred = best_model.predict(X_future)

future_df['Weekly_Sales_Pred'] = y_future_pred

# Exportar predicciones del año siguiente
future_df[['Store', 'Dept', 'Date', 'Weekly_Sales_Pred']].to_csv('predicciones_ventas_2013.csv', index=False)
print("Archivo de predicciones para el año siguiente exportado correctamente.")

# Exportar predicciones finales
predicciones_df = pd.DataFrame({
    'Store': X_test.index.map(lambda idx: ventas.iloc[idx]['Store']),
    'Dept': X_test.index.map(lambda idx: ventas.iloc[idx]['Dept']),
    'Date': X_test.index.map(lambda idx: ventas.iloc[idx]['Date']),
    'Weekly_Sales_Pred': y_pred
})

predicciones_df.to_csv('predicciones_ventas.csv', index=False)
