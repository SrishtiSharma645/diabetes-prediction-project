from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

app = Flask(__name__)

# Load and prepare data
url = "diabetes.csv"
df = pd.read_csv(url)
df = df.drop_duplicates()

# Data preprocessing
df["Glucose"] = df["Glucose"].replace(0, df["Glucose"].mean())
df["BloodPressure"] = df["BloodPressure"].replace(0, df["BloodPressure"].mean())
df["SkinThickness"] = df["SkinThickness"].replace(0, df["SkinThickness"].median())
df["Insulin"] = df["Insulin"].replace(0, df["Insulin"].median())
df["BMI"] = df["BMI"].replace(0, df["BMI"].median())

# Drop unused features
df_new = df.drop(['BloodPressure', 'Insulin', 'DiabetesPedigreeFunction'], axis=1)

# Split data
y = df_new["Outcome"]
x = df_new.drop("Outcome", axis=1)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=0)

# Train RandomForestClassifier
model = SVC(kernel='poly', C=100, probability=True)  # probability=True allows you to get probabilities
model.fit(x_train, y_train)


@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/home')
def home():
    return render_template("index.html")

@app.route('/project')
def project():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            # Get input data
            Pregnancies = int(request.form["preg"])
            Glucose = int(request.form["glu"])
            SkinThickness = int(request.form["skv"])
            BMI = float(request.form["bmi"])
            Age = int(request.form["age"])
            
            # Scale input data
            input_data = [[Pregnancies, Glucose, SkinThickness, BMI, Age]]
            input_data_scaled = scaler.transform(input_data)
            
            # Predict and get result
            prediction = model.predict(input_data_scaled)
            result = "Positive for diabetes" if prediction[0] == 1 else "Negative for diabetes"
            
            return render_template("form.html", Pregnancies=Pregnancies, Glucose=Glucose, 
                                   SkinThickness=SkinThickness, BMI=BMI, Age=Age, result=result)
        except Exception as e:
            return "Error: " + str(e)
    else:
        return "Method not allowed"

if __name__ == "__main__":
    app.run(debug=True)
