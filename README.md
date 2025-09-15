# CodeAlpha Credit Worthiness Prediction

## Project Overview
This project predicts the creditworthiness of individuals using machine learning techniques. It assesses loan eligibility based on financial and demographic features.

## Features
- Data preprocessing & feature engineering
- Machine learning model training and evaluation
- Interactive frontend using Streamlit

## Dataset
- `cs_training.csv` used for training

## Folder Structure
CodeAlpha_Credit_Worthiness_Prediction/
│
├── data/
│   ├── cs_training # Original dataset 
│   ├── cleaned_credit_data # Processed dataset
│
├── notebook/               # Jupyter notebooks for exploration and experiments
│   ├── creditworthiness_notebook.ipynb
│
├── app.py                   # Streamlit app for frontend
├── README.md

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/janhavi455/CodeAlpha_Credit_Worthiness_Prediction.git
cd CodeAlpha_Credit_Worthiness_Prediction

2. Install dependencies
pip install -r requirements.txt

3. Run the Streamlit app
streamlit run app.py
Your browser will open the app where you can input customer details and get creditworthiness predictions.

## Usage
Place your dataset (cs_training.csv) in data.
Open and run the creditworthiness_notebook to prepare the data.
Train and evaluate your model using the training notebook.
Launch the Streamlit app (app.py) to interact with your model in real-time.

## Future Improvements
Add multiple ML models and compare their performance
Implement advanced feature engineering for better predictions
Deploy the app online for public access

Author
Janhavi Sakhare
B.Tech Student | AI & Data Science Enthusiast
GitHub: janhavi455
