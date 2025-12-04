import os
import tensorflow as tf
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash,
)

from config import (
    BASE_DIR,
    MODEL_PATH,
    SECRET_KEY,
)

from inference import (
    allowed_file,
    predict_image,
    evaluate_model_on_test_set,  
)

# ---------------------------------------------------------
# Flask Initialization
# ---------------------------------------------------------
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.secret_key = SECRET_KEY

# ---------------------------------------------------------
# Load the trained model
# ---------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}. "
        f"Please run 'python training/train.py' first."
    )

print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model successfully loaded.")


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/test")
def test_page():
    return render_template("test.html")


# ---------------------------------------------------------
# Upload API — returns JSON results
# ---------------------------------------------------------
@app.route("/uploader", methods=["POST"])
def upload_file():
    files = request.files.getlist("file")
    if not files or files[0].filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("test_page"))

    results = []

    for f in files:
        if f and allowed_file(f.filename):
            try:
                label, prob_normal = predict_image(model, f)
                results.append(
                    {
                        "filename": f.filename,
                        "prediction": label,
                        "probability_normal": prob_normal,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "filename": f.filename,
                        "error": str(e),
                    }
                )
        else:
            results.append(
                {
                    "filename": f.filename,
                    "error": "Unsupported file type",
                }
            )

    return jsonify({"results": results})


# ---------------------------------------------------------
# Single-file prediction endpoint
# ---------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict_single():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        label, prob_normal = predict_image(model, f)
        return jsonify(
            {
                "filename": f.filename,
                "prediction": label,
                "probability_normal": prob_normal,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------
# Evaluation endpoint — confusion matrix, accuracy
# ---------------------------------------------------------
@app.route("/evaluate", methods=["GET"])
def evaluate():
    """
    Compute confusion matrix + accuracy on the entire test_set folder
    and return metrics in JSON format.
    """
    metrics = evaluate_model_on_test_set(model)
    return jsonify(metrics)


# ---------------------------------------------------------
# Run Server
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
