from flask import Flask
import pickle
import pandas as pd

app = Flask(__name__)

df= pd.DataFrame([['Residential',7063,76,10,'Weekday']],
       columns=[
        "Building Type",
        "Square Footage",
        "Number of Occupants",
        "Appliances Used",
        "Day of Week"])
print(df)

with open(r"R:\DATA SCIENCE\Dharmarajan\Energy_Consumation\regression.pkl", "rb") as file:
    pipeline = pickle.load(file)

print(pipeline.predict(df)[0])