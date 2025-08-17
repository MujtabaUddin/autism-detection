from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

FEATURES = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
            'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
            'age', 'gender', 'ethnicity', 'jundice', 'autism', 'contry_of_res']

@app.route('/')
def landing():
    return render_template("landing.html")  

@app.route('/check')
def check_form():
    return render_template("index.html", prediction=None, diagnosis=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        num_fields = [f"A{i}_Score" for i in range(1, 11)] + ['age']
        num_values = [float(request.form[field]) for field in num_fields]

        cat_fields = ['gender', 'ethnicity', 'jundice', 'autism', 'contry_of_res']
        cat_values = []
        for field in cat_fields:
            val = request.form[field].strip().lower()
            le = label_encoders[field]
            match = next((cls for cls in le.classes_ if cls.strip().lower() == val), None)
            if match is None:
                return render_template("index.html", prediction=f"Invalid input for {field}", diagnosis=None)
            encoded = le.transform([match])[0]
            cat_values.append(encoded)

        final_input = num_values + cat_values
        input_df = pd.DataFrame([final_input], columns=FEATURES)

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        confidence = round(max(proba) * 100, 2)

        result = "Positive for ASD" if prediction == 1  else "Negative for ASD"
        diagnosis = (
            "This screening suggests a possibility of Autism Spectrum Disorder. Further professional evaluation is recommended."
            if prediction == 1
            else "You are unlikely to show signs of Autism Spectrum Disorder based on this screening."
        )

        return render_template("index.html", prediction=f"{result} ({confidence}% confidence)", diagnosis=diagnosis)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}", diagnosis=None)

@app.route('/care')
def care():
    return render_template("care.html")

if __name__ == '__main__':
    app.run(debug=True)


