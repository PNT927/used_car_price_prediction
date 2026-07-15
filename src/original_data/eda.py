import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD DATASET
df = pd.read_csv("../../data/raw/cars.csv")

# DROP MISSING VALUES
df = df.dropna()

# USE REGEX TO EXTRACT NUMBERS FROM TEXT COLUMNS
# MILEAGE CHANGE FROM "22.4 KMPL" TO "22.4"
df['mileage'] = df['mileage'].str.extract(r'(\d+\.?\d*)').astype(float)
# ENGINE CHANGED FROM "1248 CC" TO 1248
df['engine'] = df['engine'].str.extract(r'(\d+)').astype(float)
# MAX_POWER FROM "74 BHP" TO 74
df['max_power'] = df['max_power'].str.extract(r'(\d+\.?\d*)').astype(float)

# ADDED DATASET RANGE AND SIZE FOR ANALYSIS
min_year = df["year"].min()
max_year = df["year"].max()

min_mileage = df["mileage"].min()
max_mileage = df["mileage"].max()

num_rows = df.shape[0]
num_columns = df.shape[1]

# PRINT BASIC DATASET INFO
print("Year range:", min_year, "-", max_year)
print("Mileage range:", min_mileage, "-", max_mileage)
print("Dataset size:", num_rows, "rows,", num_columns, "columns")


# 1. PRICE VS MILEAGE
plt.scatter(df['km_driven'], df['selling_price'])
plt.xlabel("Kilometers Driven")
plt.ylabel("Selling Price")
plt.title("Price vs Mileage")
plt.savefig("../../reports/figures/price_vs_mileage.png")
plt.show()

# 2. PRICE VS YEAR
plt.scatter(df['year'], df['selling_price'])
plt.xlabel("Year")
plt.ylabel("Selling Price")
plt.title("Price vs Year")
plt.savefig("../../reports/figures/price_vs_year.png")
plt.show()

# 3. CORRELATION HEATMAP
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("../../reports/figures/heatmap.png")
plt.show()


# df = pd.read_csv("../data/cars.csv")
#
# print("First 5 rows:")
# print(df.head())
#
# print("\nColumn names:")
# print(df.columns.tolist())
#
# print("\nDataset shape:")
# print(df.shape)
#
# print("\nMissing values:")
# print(df.isnull().sum())
#
# print("\nData types:")
# print(df.dtypes)