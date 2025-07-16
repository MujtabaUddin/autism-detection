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
        # Define numerical and categorical fields
        num_fields = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                      'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age']
        cat_fields = ['gender', 'ethnicity', 'jundice', 'autism', 'contry_of_res']

        # Process numerical inputs
        num_values = [float(request.form[field]) for field in num_fields]

        # Process categorical inputs with label encoders
        cat_values = []
        for field in cat_fields:
            input_val = request.form[field].strip().lower()
            le = label_encoders.get(field)

            if le is None:
                return render_template("index.html", prediction=f"Error: No encoder found for {field}.")

            matched_class = next((cls for cls in le.classes_ if cls.strip().lower() == input_val), None)
            if matched_class is None:
                return render_template("index.html", prediction=f"Error: Invalid input for {field}: '{input_val}'")

            encoded_val = le.transform([matched_class])[0]
            cat_values.append(encoded_val)

        # Combine all inputs
        final_input = num_values + cat_values
        input_df = pd.DataFrame([final_input], columns=FEATURES)

        # Make prediction
        prediction = model.predict(input_df)[0]

        # (Optional) For models with predict_proba
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[0]
            confidence = round(np.max(prob) * 100, 2)
        else:
            confidence = None

        # Human-readable result
        if isinstance(prediction, str):
            result = "Positive for ASD" if prediction.strip().lower() == "yes" else "Negative for ASD"
        else:
            result = "Positive for ASD" if prediction == 1 else "Negative for ASD"

        # Add confidence if available
        if confidence is not None:
            result += f" (Confidence: {confidence}%)"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
