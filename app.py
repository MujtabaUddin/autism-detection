from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model and label encoders
model = pickle.load(open("model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

# Final features used during training
FEATURES = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
            'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
            'age', 'gender', 'ethnicity', 'jundice', 'autism', 'contry_of_res']

@app.route('/')
def home():
    return render_template("index.html", prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Numerical fields
        num_fields = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                      'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age']
        num_values = [float(request.form[field]) for field in num_fields]

        # Categorical fields
        cat_fields = ['gender', 'ethnicity', 'jundice', 'autism', 'contry_of_res']
        cat_values = []

        for field in cat_fields:
            input_val = request.form[field].strip().lower()
            le = label_encoders[field]

            matched_class = next((cls for cls in le.classes_ if cls.strip().lower() == input_val), None)
            if matched_class is None:
                return render_template("index.html", prediction=f"Error: Invalid input for {field}: '{input_val}'")

            encoded_val = le.transform([matched_class])[0]
            cat_values.append(encoded_val)

        # Combine features
        final_input = num_values + cat_values
        final_input_df = pd.DataFrame([final_input], columns=FEATURES)

        # Predict
        prediction = model.predict(final_input_df)[0]

        # Handle both string and numeric outputs
        if isinstance(prediction, str):
            result = "Positive for ASD" if prediction.strip().lower() == "yes" else "Negative for ASD"
        else:
            result = "Positive for ASD" if prediction == 1 else "Negative for ASD"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
