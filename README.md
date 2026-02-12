# 🏠 Germany House Price Prediction App
## 📌 Overview
The Germany House Price Prediction App is an interactive machine learning application designed to estimate rental prices for apartments across Germany.

Built with Streamlit and Scikit-Learn, the app allows users to input property details (such as living space, number of rooms, and location) to receive an instant rent estimation. The project also includes scripts for training the model and performing market analysis on German housing data.

## 🚀 Features
Interactive Web Interface: User-friendly form to input apartment details.

Real-time Prediction: Instant estimation of "Total Rent" based on a trained Random Forest model.

Comparison Tool: Compare the predicted rent against a known actual rent to gauge deal quality.

Market Analysis Utilities: Includes functions to calculate ROI, rent increases per ZIP code, and heating type distribution.

Automated Training Pipeline: Script to clean data, train the model, and save performance metrics.

## 📂 Project Structure
Plaintext
**Germany_Hause_Price_Prediction_App**/
├── data/
│   └── immo_data.csv          # Dataset (input for training)
├── model/
│   ├── housing_model.pkl      # Trained Random Forest model (generated)
│   └── metrics.json           # Model performance scores (generated)
├── app.py                     # Main Streamlit application
├── train_model.py             # Script to train and save the model
├── functions.py               # Data analysis helper functions (ROI, trends)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
## 🛠️ Tech Stack
Frontend: Streamlit

**Machine Learning**: Scikit-Learn (RandomForestRegressor)

**Data Manipulation:** Pandas, NumPy

**Persistence:** Joblib
