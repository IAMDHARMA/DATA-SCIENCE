🏢 Energy Consumption Prediction API

This project provides a Flask-based REST API to predict energy consumption for different buildings based on key factors like building type, size, number of occupants, and appliance usage. It uses a trained machine learning regression model (regression.pkl) for predictions.

🚀 Features

Predict energy consumption using machine learning.

RESTful endpoints built with Flask.

Handles JSON input for predictions.

Easy integration with front-end or external systems.

📂 Project Structure
Energy_Consumption_Project/
│
├── app.py                   # Flask API script
├── Energy_consumation.ipynb # Jupyter notebook for model building and training
├── regression.pkl           # Trained machine learning regression model
└── README.md                # Project documentation

🔧 Installation
1. Clone the repository
git clone <your-repo-url>
cd Energy_Consumption_Project

2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

3. Install dependencies
pip install -r requirements.txt


Note: If requirements.txt is not available, install manually:

pip install flask pandas scikit-learn

▶️ Running the API
Start the Flask server:
python app.py


The server will run at:

http://127.0.0.1:5000

🌐 API Endpoints
1. Health Check

Check if the API is running.

URL: /hi

Method: GET

Response Example:

{
  "message": "hello world",
  "status": 200,
  "male": "male"
}

2. Predict Energy Consumption

Make a prediction by sending building data.

URL: /predict

Method: POST

Request Body (JSON):

{
  "Building Type": "Residential",
  "Square Footage": 1200,
  "Number of Occupants": 4,
  "Appliances Used": 15,
  "Day of Week": "Monday"
}


Response Example:

{
  "prediction": 350.75
}

🧠 Model Information

The model (regression.pkl) is a trained machine learning pipeline built using Scikit-learn.

Input Features:

Building Type

Square Footage

Number of Occupants

Appliances Used

Day of Week

🛠 Tools and Technologies

Backend: Flask

Data Analysis & Model Training: Pandas, Scikit-learn

Language: Python

Deployment Ready: Can be deployed to AWS, Heroku, or any cloud platform.

📜 License

This project is licensed under the MIT License. You are free to use and modify it.
