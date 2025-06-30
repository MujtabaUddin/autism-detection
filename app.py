from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html", prediction=None)

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Numerical features from form
        num_fields = ['A1_Score','A2_Score','A3_Score','A4_Score','A5_Score',
                      'A6_Score','A7_Score','A8_Score','A9_Score','A10_Score',
                      'age', 'result']
        num_values = [float(request.form[field]) for field in num_fields]

        # Categorical features from form
        cat_fields = ['gender', 'ethnicity', 'jundice', 'autism', 'contry_of_res']
        cat_values = []
        for field in cat_fields:
            val = request.form[field].lower().strip()
            le = label_encoders[field]
            if val not in le.classes_:
                return render_template("index.html", prediction=f"Error: Invalid input for {field}: '{val}'")
            encoded_val = le.transform([val])[0]
            cat_values.append(encoded_val)

        # Combine and predict
        final_input = np.array(num_values + cat_values).reshape(1, -1)
        prediction = model.predict(final_input)[0]
        result = "Positive for ASD" if prediction == 1 else "Negative for ASD"
        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)

