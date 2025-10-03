from flask import Flask, request
import pandas as pd
import pickle

app = Flask(__name__)

@app.post("/MLpredict")
def MLmodel_predict():
    data = request.get_json()
    
    if not data:
        return {"error": "No JSON data received"}
    
    sepal_length = data.get("sepal.length")
    sepal_width = data.get("sepal.width")
    petal_length = data.get("petal.length")
    petal_width = data.get("petal.width")

    with open(r"R:\DATA SCIENCE\DEEP LEARNING PROJECTS\Iris flower Prediction Project\iris flower ML.pkl", "rb") as file:
        Model = pickle.load(file)
    
    df = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=["sepal.length", "sepal.width", "petal.length", "petal.width"]
    )

    prediction = Model.predict(df)

    return {"prediction": float(prediction[0])}


@app.post("/DLpredict")
def DLmodel_predict():
    data = request.get_json()

    if not data:
        return {"error": "No JSON data received"}

    sepal_length = data.get("sepal.length")
    sepal_width = data.get("sepal.width")
    petal_length = data.get("petal.length")
    petal_width = data.get("petal.width")

    with open(r"R:\DATA SCIENCE\DEEP LEARNING PROJECTS\Iris flower Prediction Project\iris_model_DL.pkl", "rb") as file:
        DLModel = pickle.load(file)

    df = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=["sepal.length", "sepal.width", "petal.length", "petal.width"]
    )

    prediction_DL = DLModel.predict(df)

    return {"prediction": float(prediction_DL[0])}


if __name__ == "__main__":
    app.run(debug=True)
