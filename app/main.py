from flask import Flask, render_template, request, session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = "car_price_project_secret_key"


# -----------------------
# LOAD AND PREPARE UK DATA
# -----------------------
df = pd.read_csv("../data/raw/used_cars_UK.csv")

df = df.drop(columns=["Unnamed: 0", "Service history"], errors="ignore")

df = df.rename(columns={
    "Price": "selling_price",
    "Mileage(miles)": "mileage",
    "Registration_Year": "year",
    "Fuel type": "fuel",
    "Gearbox": "transmission"
})

df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")
df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")

df = df.dropna(subset=["selling_price", "mileage", "year", "title"])

# Extract make and model
df["make"] = df["title"].str.split().str[0]
df["model"] = df["title"].str.split().str[1]




make_model_map = (
    df.groupby("make")["model"]
    .apply(lambda x: sorted(x.dropna().unique().tolist()))
    .to_dict()
)

valid_options_map = {}

for _, row in df.iterrows():
    make = row["make"]
    model_name = row["model"]
    fuel = row["fuel"]
    transmission = row["transmission"]

    if make not in valid_options_map:
        valid_options_map[make] = {}

    if model_name not in valid_options_map[make]:
        valid_options_map[make][model_name] = {
            "fuels": set(),
            "transmissions": set()
        }

    valid_options_map[make][model_name]["fuels"].add(fuel)
    valid_options_map[make][model_name]["transmissions"].add(transmission)

# convert sets to sorted lists so HTML/JS can use them
for make in valid_options_map:
    for model_name in valid_options_map[make]:
        valid_options_map[make][model_name]["fuels"] = sorted(
            list(valid_options_map[make][model_name]["fuels"])
        )
        valid_options_map[make][model_name]["transmissions"] = sorted(
            list(valid_options_map[make][model_name]["transmissions"])
        )

make_list = sorted(make_model_map.keys())


# FEATURE ENGINEERING
X = df[['year', 'mileage', 'fuel', 'transmission','make', 'model']]
y = df['selling_price']

X = pd.get_dummies(X, drop_first=True)
# Save training columns so user input matches
model_columns = X.columns


# TRAIN MODEL
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
residuals = y_test - y_pred_test
threshold = residuals.std()
@app.route("/")
def home():
    return render_template("home.html", makes=make_list,make_model_map=make_model_map,  valid_options_map=valid_options_map)


@app.route("/predict", methods=["POST"])
def predict():
    year = int(request.form["year"])
    mileage = int(request.form["mileage"])
    actual_price = float(request.form["actual_price"])
    fuel = request.form["fuel"]
    transmission = request.form["transmission"]
    car_model = request.form["model"]
    make = request.form["make"]

    input_data = pd.DataFrame([{
        "year": year,
        "mileage": mileage,
        "fuel": fuel,
        "transmission": transmission,
        "make": make,
        "model": car_model
    }])

    input_data = pd.get_dummies(input_data, drop_first=True)
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_data)[0]
     # Calculate residual
    residual = actual_price - prediction

    abs_difference = abs(residual)

    if residual > threshold:
        anomaly = "Overpriced"
        status_label = "This car looks overpriced"

    elif residual < -threshold:
        anomaly = "Underpriced"
        status_label = "This car looks underpriced"

    else:
        anomaly = "Fairly Priced"
        status_label = "This car is priced close to fair value"



    if residual > 0:
        price_direction = "above"
    elif residual < 0:
        price_direction = "below"
    else:
        price_direction = "equal to"



    explanation_points = [
        f"The estimated fair price is around £{prediction:,.2f}.",
        f"The listed price is £{abs_difference:,.2f} {price_direction} this estimate.",
        "This estimate is based on similar cars using year, mileage, fuel type, transmission, make, and model."
    ]

    session["latest_result"] = {
        "prediction": round(prediction, 2),
        "actual_price": round(actual_price, 2),
        "residual": round(residual, 2),
        "abs_difference": round(abs_difference, 2),
        "anomaly": anomaly,
        "threshold": round(threshold, 2),
        "year": year,
        "make": make,
        "car_model": car_model,
        "mileage": mileage,
        "fuel": fuel,
        "transmission": transmission
    }

    return render_template(
        "result.html",
        prediction=round(prediction, 2),
        actual_price=round(actual_price, 2),
        residual=round(residual, 2),
        abs_difference=round(abs_difference, 2),
        anomaly=anomaly,
        threshold=round(threshold, 2),
        status_label=status_label,
        price_direction=price_direction,
        explanation_points=explanation_points,
        year=year,
        make=make,
        car_model=car_model,

    )

@app.route("/insights")
def insights():
    latest_result = session.get("latest_result")

    if latest_result is None:
        return render_template(
            "home.html",
            makes=make_list,
            make_model_map=make_model_map,
            valid_options_map=valid_options_map
        )
    return render_template("insights.html",result=latest_result)

if __name__ == "__main__":
    app.run(debug=True)

    # --------------- New work - Not possible due to Time constraint ----------------------------
    # if abs_difference <= threshold * 0.5:
    #     confidence_text = "High confidence: price is very close to estimate."
    # elif abs_difference <= threshold:
    #     confidence_text = "Moderate confidence: price is reasonably close."
    # else:
    #     confidence_text = "Lower confidence: price is far from estimate."