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
```plaintext
CodeAlpha_Credit_Worthiness_Prediction/
│
├── data/
│   ├── cs_training              # Original dataset
│   ├── cleaned_credit_data      # Processed dataset
│
├── models/                      
│   ├── creditworthiness_model.pkl         # Saved trained model
│
├── notebook/                    
│   ├── creditworthiness_notebook.ipynb   # Jupyter notebook for exploration and experiments
│
├── app.py                       # Streamlit app for frontend
├── requirements.txt
├── README.md
```

## Installation<br>

1. **Clone the repository**
```bash
git clone https://github.com/janhavi455/CodeAlpha_Credit_Worthiness_Prediction.git
cd CodeAlpha_Credit_Worthiness_Prediction
```

2. **Install dependencies** 
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**  
```
streamlit run app.py
```
Your browser will open the app where you can input customer details and get creditworthiness predictions.

## Usage
Place your dataset (cs_training.csv) in data.<br>
Open and run the creditworthiness_notebook to prepare the data.<br>
Train and evaluate your model using the training notebook.<br>
Launch the Streamlit app (app.py) to interact with your model in real-time.<br>

## Future Improvements
Add multiple ML models and compare their performance.<br>
Implement advanced feature engineering for better predictions.<br>
Deploy the app online for public access.<br>

## **Author**<br>
Janhavi Sakhare<br>
B.Tech Student | AI & Data Science Enthusiast<br>
GitHub: janhavi455<br>
