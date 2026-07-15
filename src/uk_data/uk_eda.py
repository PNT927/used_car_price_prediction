import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("../../data/raw/used_cars_UK.csv")

# Same cleaning as before
df = df.drop(columns=["Unnamed: 0", "Service history"], errors="ignore")

df = df.rename(columns={
    "Price": "selling price",
    "Mileage(miles)": "mileage",
    "Registration_Year": "year",
    "Fuel type": "fuel",
    "Gearbox": "transmission"
})

df = df.dropna(subset=["selling price", "mileage", "year"])

# added dataset range and size for analysis
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


# __________________GRAPHS________________________

# Price vs Mileage
plt.figure()
sns.scatterplot(x='mileage', y='selling price', data=df)
plt.xlabel("Mileage (miles)")
plt.ylabel("Selling Price (£)")
plt.title("UK Dataset: Price vs Mileage")
plt.savefig("../../reports/figures/uk_price_vs_mileage.png")
plt.show()

# Price vs Year
# use seaborn for better styling
plt.figure()
sns.scatterplot(x='year', y='selling price', data=df)
plt.xlabel("Year")
plt.ylabel("Selling Price (£)")
plt.title("UK Dataset: Price vs Year")
plt.savefig("../../reports/figures/uk_price_vs_year.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("UK Dataset Correlation Heatmap")
plt.savefig("../../reports/figures/uk_heatmap.png")
plt.show()



# Histogram
# Where most cars are concentrated (check if values are evenly spread)

# Price distribution
plt.figure()
plt.hist(df['selling price'], bins=30)
plt.xlabel("Price (£)")
plt.ylabel("Number of Cars")
plt.title("Car Prices Distribution")
plt.savefig("../../reports/figures/uk_price_distribution.png")
plt.show()

# Mileage distribution
plt.figure()
plt.hist(df['mileage'] / 1000, bins=30)
plt.xlabel("Mileage (thousands of miles)")
plt.ylabel("Number of Cars")
plt.title("Mileage Distribution")
plt.savefig("../../reports/figures/uk_mileage_distribution.png")
plt.show()

# Year distribution
plt.figure()
plt.hist(df['year'], bins=20)
plt.xlabel("Year")
plt.ylabel("Number of Cars")
plt.title("Years Distribution")
plt.savefig("../../reports/figures/uk_year_distribution.png")
plt.show()




# Skewness + spread analysis
print("\n------- DISTRIBUTION INSIGHTS --------")

# Positive skew = right = lots of cheap cars, fewer expensive one
# Negative skew = left = opposite
print(f"Price skewness: {df['selling price'].skew():.2f}")
print(f"Mileage skewness: {df['mileage'].skew():.2f}")
print(f"Year skewness: {df['year'].skew():.2f}")


# How wide the data is and how much variation exist
print("\nSpread (standard deviation):")
print(f"Price std: {df['selling price'].std():.2f}")
print(f"Mileage std: {df['mileage'].std():.2f}")
print(f"Year std: {df['year'].std():.2f}")








# print("FIRST 5 ROWS:\n", df.head())
# print("\nCOLUMNS:\n", df.columns.tolist())
# print("\nSHAPE:\n", df.shape)
# print("\nMISSING VALUES:\n", df.isnull().sum())
# print("\nDATA TYPES:\n", df.dtypes)