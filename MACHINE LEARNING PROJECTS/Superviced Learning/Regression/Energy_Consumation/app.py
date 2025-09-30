from flask import Flask, request
import pandas as pd
import pickle

app = Flask(__name__)

@app.get("/hi")
def hello():
    return {
        "message": "hello world",
        "status": 200,
        "male": "male"
    }

@app.post("/predict")
def model_predict():
    # FIX: Call get_json() correctly
    data = request.get_json()

    if not data:
        return {"error": "No JSON data received"}, 400

    # Extract values from the JSON body safely
    Building_Type = data.get("Building Type")
    Square_Footage = data.get("Square Footage")
    Number_of_Occupant = data.get("Number of Occupants")
    Appliances_Used = data.get("Appliances Used")
    Day_of_Week = data.get("Day of Week")

    # Load your trained model
    with open(r"R:\DATA SCIENCE\Dharmarajan\Energy_Consumation\regression.pkl", "rb") as file:
        pipeline = pickle.load(file)

    # Create dataframe for prediction
    df = pd.DataFrame(
        [[Building_Type, Square_Footage, Number_of_Occupant, Appliances_Used, Day_of_Week]],
        columns=[
            "Building Type",
            "Square Footage",
            "Number of Occupants",
            "Appliances Used",
            "Day of Week"
        ]
    )

    # Make prediction
    prediction = pipeline.predict(df)

    return {
        "prediction": float(prediction[0])  # Convert to float to make it JSON serializable
    }

if __name__ == "__main__":
    app.run(debug=True)
