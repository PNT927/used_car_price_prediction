import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score


# load dataset
df = pd.read_csv("../../data/raw/cars.csv")

df = df.dropna()

# USE REGEX TO EXTRACT NUMBERS FROM TEXT COLUMNS
# MILEAGE CHANGE FROM "22.4 KMPL" TO "22.4"
df['mileage'] = df['mileage'].str.extract(r'(\d+\.?\d*)').astype(float)
# ENGINE CHANGED FROM "1248 CC" TO 1248
df['engine'] = df['engine'].str.extract(r'(\d+)').astype(float)
# MAX_POWER FROM "74 BHP" TO 74
df['max_power'] = df['max_power'].str.extract(r'(\d+\.?\d*)').astype(float)

# Define features X and target y
# FEATURE ENGINEERING
X = df[['year', 'km_driven', 'engine', 'max_power', 'mileage',
        'fuel', 'transmission', 'owner']]
y = df['selling_price']

X = pd.get_dummies(X, drop_first=True)


# SPLIT THE DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining set size:", X_train.shape)
print("Test set size:", X_test.shape)

# MODEL DEVELOPMENT
model = LinearRegression()
# TRAIN MODEL
model.fit(X_train, y_train)
print("\nModel trained successfully")

# MAKE PREDICTIONS
y_pred = model.predict(X_test)
print("\nPredictions:")
print(y_pred[:5])

#EVAULATINGT THE MODEL
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nLinear Regression Results:")
print("Mean Absolute Error (MAE):", mae)
print("R2 Score:", r2)


# Anomaly Detection

# CALCULATE RESIDUALS
df_test = X_test.copy()
df_test['actual_price'] = y_test
df_test['predicted_price'] = y_pred
df_test['residual'] = df_test['actual_price'] - df_test['predicted_price']

threshold = df_test['residual'].std()

# FLAG ANOMALIES
df_test['anomaly'] = df_test['residual'].apply(
    lambda x: 'Overpriced' if x > threshold else ('Underpriced' if x < -threshold else 'Normal'))
# SHOW RESULTS
print("\nAnomaly Detection Results:")
print(df_test[['actual_price', 'predicted_price', 'residual', 'anomaly']].head(10))



# RIDGE REGRESSION

ridge_model = Ridge(alpha=1.0)

ridge_model.fit(X_train, y_train)

y_pred_ridge = ridge_model.predict(X_test)

# Evaluate
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("\nRidge Regression Results:")
print("MAE:", mae_ridge)
print("R2 Score:", r2_ridge)


# DATASET SIZE
print("\n==================================================================")
print("==================================================================")
print("\nNumber of rows:", df.shape[0])
print("Number of features:", df.shape[1])

print("Min year:", df['year'].min())
print("Max year:", df['year'].max())

print("Min mileage:", df['mileage'].min())
print("Max mileage:", df['mileage'].max())