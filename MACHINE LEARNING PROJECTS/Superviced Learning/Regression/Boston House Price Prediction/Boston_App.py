from flask import Flask,request
import pandas as pd
import pickle

app=Flask(__name__)

@app.post("/prediction")
def model_predict():
    data=request.get_json()

    if not data:
        return{"error":"No Json data received",
    
    CRIM=data.get("CRIM"),	
    ZN	= data.get("ZN"),
    INDUS = data.get("INDUS"),
    NOX	= data.get("NOX"),
    RM	= data.get("RM"),
    AGE	=data.get("AGE"),
    RAD	=data.get("RAD"),
    TAX	=data.get("TAX"),
    PTRATIO = data.get("PTRATIO"),
    B=data.get("B"),
    LSTAT=data.get("LSTAT")
        }

with open(r"R:\DATA SCIENCE\Dharmarajan\Project_1_Boston House Price Prediction\Boston House Price Prediction.pkl","rb")as file:
    pipeline=pickle.load(file)

df=pd.DataFrame([[CRIM,ZN,INDUS,NOX,RM,AGE,RAD,TAX,PTRATIO,B,LSTAT]],
                columns=[
                    "CRIM"
                    "ZN"
                    "INDUS"
                    "NOX"
                    "RM"
                    "AGE"
                    "RAD"
                    "TAX"
                    "PTRATIO"
                    "B"
                    "LSTAT"
                ])
prediction = pipeline.predict(df)