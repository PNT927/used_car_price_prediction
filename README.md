# Used Car Price Prediction System

## Overview
This project is a machine learning-based web application that predicts the fair price of used cars and identifies whether listings are overpriced or underpriced.

## Features
- Price prediction using Linear Regression
- Anomaly detection (Overpriced / Underpriced / Normal)
- Feature engineering (year, mileage, fuel, transmission, make, model)
- Interactive web application using Flask

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Flask (web app)

## Project Structure
- `src/` – data processing and model scripts
- `app/` – Flask web application
- `data/` – datasets
- `reports/figures/` – graphs and visual outputs
- `models/` – saved models (future)

### How to Run

## Installation & Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Run the application
```bash
python app/main.py
```

### 3. Open in browser
http://127.0.0.1:5000/

Usage
Enter vehicle details such as mileage, year, fuel type, and transmission
Submit the form
View the predicted price and classification result:
Underpriced
Fairly Priced
Overpriced
Dataset

The project uses publicly available used car datasets, including a UK-specific dataset to better reflect real-world market conditions. The data was cleaned and preprocessed to remove inconsistencies and ensure suitability for modelling.




## Philip Tagoe Jr
### Final Year Computer Science Project
Candidate Number: 267109
