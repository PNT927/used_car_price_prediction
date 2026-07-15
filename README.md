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

## Usage

1. Enter vehicle details such as mileage, year, fuel type, transmission, make, and model.
2. Enter the listed price for comparison.
3. Submit the form.
4. View the predicted fair price and classification:
   - Underpriced
   - Fairly Priced
   - Overpriced

## Dataset

The project uses publicly available used car datasets, including a UK-specific dataset to better reflect real-world market conditions. The data was cleaned and preprocessed to remove missing values, inconsistent formats, and unsuitable records before modelling.

## Model

Multiple Linear Regression was used to estimate used car prices based on features including year, mileage, fuel type, transmission, make, and model.

The final UK model achieved:

- Mean Absolute Error: approximately £1,301
- R² score: approximately 0.793

Residual analysis was used to compare the predicted price with the listed price and classify each listing as underpriced, fairly priced, or overpriced.

## Limitations

- Vehicle condition, accident history, service history, and location are not fully represented.
- The system uses static datasets rather than live marketplace data.
- Linear Regression may not capture every complex or non-linear relationship.
- Predictions should be treated as estimates rather than guaranteed market values.

## Future Work

- Compare Random Forest and Gradient Boosting models
- Add real-time marketplace data collection
- Introduce regional price analysis
- Develop vehicle recommendation features
- Add interactive dashboards
- Improve anomaly detection methods

## Author

Philip Tagoe Jr  
Final-Year BSc Computer Science Project  
Candidate Number: 267109