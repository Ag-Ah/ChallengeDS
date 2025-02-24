import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware

# -------------------- Configurar la API --------------------
app = FastAPI()

# Permitir conexiones desde cualquier origen (para pruebas locales)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Cargar el Modelo --------------------
try:
    model = joblib.load("modelo_ventas_xgb_optimizado.pkl")
    print("Modelo cargado correctamente.")
except FileNotFoundError:
    print("Error: El archivo 'modelo_ventas_xgb_optimizado.pkl' no se encontró.")

# -------------------- Definir el Esquema de Datos de Entrada --------------------

class SalesPredictionRequest(BaseModel):
    Store: int
    Dept: int
    Size: float
    Temperature: float
    Fuel_Price: float
    CPI: float
    Unemployment: float
    IsHoliday: bool
    Year: int
    Month: int
    Week: int
    Total_MarkDown: float
    Sales_Last_Year: float
    MarkDown1: float
    MarkDown2: float
    MarkDown3: float
    MarkDown4: float
    MarkDown5: float
    DiscountImpact: float
    Size_Discount_Interaction: float
    Unemployment_Sales_Interaction: float
    
# -------------------- Endpoint: Predicción de las Próximas Cuatro Semanas --------------------
import pandas as pd

@app.post("/predict_next_four_weeks")
def predict_sales(request: SalesPredictionRequest):
    print("Datos recibidos:", request.dict())
    predictions = []
    current_date = datetime.now()

    for week_offset in range(1, 5):
        date = current_date + timedelta(weeks=week_offset)

        data_dict = {
            "Store": int(request.Store),
            "Dept": int(request.Dept),
            "Size": float(request.Size),
            "Temperature": float(request.Temperature),
            "Fuel_Price": float(request.Fuel_Price),
            "CPI": float(request.CPI),
            "Unemployment": float(request.Unemployment),
            "IsHoliday": int(request.IsHoliday),
            "Year": int(date.year),
            "Month": int(date.month),
            "Week": int(date.isocalendar().week),
            "Total_MarkDown": float(request.Total_MarkDown),
            "Sales_Last_Year": float(request.Sales_Last_Year),
            "MarkDown1": float(request.MarkDown1),
            "MarkDown2": float(request.MarkDown2),
            "MarkDown3": float(request.MarkDown3),
            "MarkDown4": float(request.MarkDown4),
            "MarkDown5": float(request.MarkDown5),
            "DiscountImpact": float(request.DiscountImpact),
            "Size_Discount_Interaction": float(request.Size_Discount_Interaction),
            "Unemployment_Sales_Interaction": float(request.Unemployment_Sales_Interaction)
        }

        df = pd.DataFrame([data_dict])
        print("DataFrame de entrada:", df)

        # Convertir la predicción a tipo float de Python
        prediction = float(model.predict(df)[0])
        predictions.append({"date": date.strftime('%Y-%m-%d'), "predicted_sales": prediction})

    return {"predictions": predictions}

# -------------------- Mensaje de Confirmación --------------------
print("API lista para recibir solicitudes.")