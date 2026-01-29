from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


model=joblib.load("Logistic_heart.pkl")
scaler=joblib.load("scaler1.pkl")
columns = scaler.feature_names_in_

app = FastAPI()

class HeartInput(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int  
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str


@app.post("/predict")
def predict_heart_disease(data:HeartInput):

       input_df=pd.DataFrame([[0]*len(columns)],columns=columns)   

       input_df["Age"]=data.Age
       
     
       input_df["RestingBP"]=data.RestingBP
       input_df["Cholesterol"]=data.Cholesterol
                
       input_df["MaxHR"]=data.MaxHR
       input_df["Oldpeak"]=data.Oldpeak


       input_df["ismale"]=1 if data.Sex=="M" else 0
       input_df["FastingBS"]=data.FastingBS
       input_df["ExerciseAngina"]=1 if data.ExerciseAngina=="Y" else 0

       cp=f"ChestPainType_{data.ChestPainType}"
       if cp in input_df.columns:
           input_df[cp]=1

       ecg=f"RestingECG_{data.RestingECG}"
       if ecg in input_df.columns:
           input_df[ecg]=1

       slope=f"ST_Slope_{data.ST_Slope}"
       if slope in input_df.columns:
           input_df[slope]=1
      
       scaled=scaler.transform(input_df)

       pred=model.predict(scaled)[0]
       prob=model.predict_proba(scaled)[0][1]
       

       return {"Prediction":int(pred),"Probability of Heart Disease":float(prob)}