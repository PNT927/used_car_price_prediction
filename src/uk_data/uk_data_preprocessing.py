import pandas as pd

df = pd.read_csv("../../data/raw/used_cars_UK.csv")

# Drop useless columns
df = df.drop(columns=["Unnamed: 0", "Service history"], errors="ignore")

# Rename columns to match your main dataset
df = df.rename(columns={
    "Price": "selling price",
    "Mileage(miles)": "mileage",
    "Registration_Year": "year",
    "Fuel type": "fuel",
    "Gearbox": "transmission"
})

# Convert types
df["selling_price"] = pd.to_numeric(df["selling price"], errors="coerce")
df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")

# Drop missing core values
df = df.dropna(subset=["selling price", "mileage", "year"])



print("CLEANED DATA:\n", df.head())
print("\nSHAPE:", df.shape)
print("\nSUMMARY OF NUMERICAL DATA", df.describe())


