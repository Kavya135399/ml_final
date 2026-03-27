from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load model & scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))   # save this also!

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def prediction():
    return render_template("predict.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')
# 🔹 Manual input prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final = scaler.transform([features])

        prediction = model.predict(final)[0]

        result = "Cancerous (Malignant)" if prediction == 1 else "Non-Cancerous (Benign)"

        return render_template('index.html', prediction_text=result)

    except:
        return render_template('index.html', prediction_text="Invalid Input")

# 🔹 CSV Upload Prediction
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    
    if not file:
        return "No file uploaded"

    data = pd.read_csv(file)

    print("Before cleaning:", data.shape)

    # ❌ Drop unwanted columns
    data = data.drop(['id'], axis=1, errors='ignore')
    data = data.drop(['diagnosis'], axis=1, errors='ignore')

    print("After cleaning:", data.shape)

    # ✅ Ensure only 30 features
    if data.shape[1] != 30:
        return f"Error: Expected 30 features, got {data.shape[1]}"

    # ✅ Convert to float
    data = data.astype(float)

    # ✅ Scale
    data_scaled = scaler.transform(data)

    # ✅ Predict
    predictions = model.predict(data_scaled)

    data['Prediction'] = predictions

    return data.to_html()

if __name__ == "__main__":
    app.run(debug=True)