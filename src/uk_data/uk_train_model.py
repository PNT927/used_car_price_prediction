import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score


# LOAD DATA

df = pd.read_csv("../../data/raw/used_cars_UK.csv")


# CLEAN DATA

df = df.drop(columns=["Unnamed: 0", "Service history"], errors="ignore")

df = df.rename(columns={
    "Price": "selling_price",
    "Mileage(miles)": "mileage",
    "Registration_Year": "year",
    "Fuel type": "fuel",
    "Gearbox": "transmission"
})

# Convert to numeric
df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")
df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")

df = df.dropna(subset=["selling_price", "mileage", "year"])
df['make'] = df['title'].str.split().str[0]
df['model'] = df['title'].str.split().str[1]


# FEATURE ENGINEERING

X = df[['year', 'mileage', 'fuel', 'transmission', 'make', 'model']]
y = df['selling_price']

# Convert categorical variables
X = pd.get_dummies(X, drop_first=True)


# TRAIN TEST SPLIT
# Random seed is used so the train/test split is reproducible.
# This means the same rows are used for training and testing each time the program runs.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining set size:", X_train.shape)
print("Test set size:", X_test.shape)


# LINEAR
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nUK Linear Regression:")
print("MAE:", mae)
print("R2:", r2)


# ANOMALY DETECTION (UK)

# Create test dataframe
df_test = X_test.copy()
df_test['actual_price'] = y_test
df_test['predicted_price'] = y_pred
df_test['residual'] = df_test['actual_price'] - df_test['predicted_price']

# Dynamic threshold
threshold = df_test['residual'].std()


# Flag anomalies
df_test['anomaly'] = df_test['residual'].apply(
    lambda x: 'Overpriced' if x > threshold else ('Underpriced' if x < -threshold else 'Normal')
)

print("\nUK Anomaly Detection Results:")
print(df_test[['actual_price', 'predicted_price', 'residual', 'anomaly']].head(10))


# RIDGE
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

y_pred_ridge = ridge_model.predict(X_test)

mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("\nUK Ridge Regression:")
print("MAE:", mae_ridge)
print("R2:", r2_ridge)

