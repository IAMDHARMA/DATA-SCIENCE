from flask import Flask,request
import pandas as pd
import pickle

app = Flask(__name__)


@app.post("/predict")
def model_predict():
    data=request.get_json()

    if not data:
        return {"error":"no Json data received"}, 400
    
    sepal_length=data.get("sepal.length")
    sepal_width=data.get("sepal.width")
    petal_length=data.get("petal.length")
    petal_width=data.get("petal.width")


    with open(r"R:\DATA SCIENCE\Dharmarajan\IRIS FLOWER PREDICTION CLASSIFICATION\iris flower.pkl","rb") as f:
        pipe_for_ran=pickle.load(f)
    
    df=pd.DataFrame(
        [[sepal_length,sepal_width,petal_length,petal_width]],
        columns=[
            "sepal.length",
            "sepal.width",
            "petal.length",
            "petal.width"        
            ]
    )
    prediction=pipe_for_ran.predict(df)

    return{
        "prediction":float(prediction[0])
    }

if __name__=="__main__":
    app.run(debug=True)