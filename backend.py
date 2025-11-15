from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Trained Models")

print("Loading model and vectorizer...")
vectorizer = joblib.load(os.path.join(MODEL_DIR, 'fake_news_vectorizer.pkl'))
model = joblib.load(os.path.join(MODEL_DIR, 'fake_news_model.pkl'))
print("Model and vectorizer loaded successfully!")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_input = request.form.get("user_input", "").strip()
        if not user_input:
            return jsonify({"error": "No input provided"}), 400

        X_test_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(X_test_tfidf)[0]
        type_of_email = "True" if prediction == 1 else "Fake"

        return jsonify({"Email_Type": type_of_email})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
