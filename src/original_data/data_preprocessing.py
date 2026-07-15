import pandas as pd

# load dataset
df = pd.read_csv("../../data/raw/cars.csv")

# drop missing values
df = df.dropna()

# USE REGEX TO EXTRACT NUMBERS FROM TEXT COLUMNS
# MILEAGE CHANGE FROM "22.4 KMPL" TO "22.4"
df['mileage'] = df['mileage'].str.extract(r'(\d+\.?\d*)').astype(float)
# ENGINE CHANGED FROM "1248 CC" TO 1248
df['engine'] = df['engine'].str.extract(r'(\d+)').astype(float)
# MAX_POWER FROM "74 BHP" TO 74
df['max_power'] = df['max_power'].str.extract(r'(\d+\.?\d*)').astype(float)



# CHECK CLEANED DATA
# SHOW THE FIRST 5 ROWS AND PRINT THE DATA TYPE
print("Cleaned data:")
print(df.head())
print("\nNew data types:")
print(df.dtypes)


