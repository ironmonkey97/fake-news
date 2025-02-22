from flask import Flask, render_template, request
import pickle

# Load the model and vectorizer with correct filenames
model = pickle.load(open("optimized_rf_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get text input from form
        news_text = request.form.get("news_text", "").strip()

        # Check if a file was uploaded
        file = request.files.get("file_input")
        if file and file.filename.endswith(".txt"):
            news_text = file.read().decode("utf-8").strip()

        # Ensure there's input to classify
        if not news_text:
            return render_template("index.html", prediction_text="Error: No input provided.")

        # Transform input and make prediction
        text_vectorized = vectorizer.transform([news_text])
        prediction = model.predict(text_vectorized)[0]

        # Determine result
        result = "Fake News" if prediction == 1 else "Real News"

        return render_template("index.html", prediction_text=f"Prediction: {result}")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
