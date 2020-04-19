from flask import Flask
from flask import request, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

VECTORIZER_PATH = 'text_analysis_vectorizer_1587325013.joblib'
MODEL_PATH = 'text_analysis_classifier_1587325013.joblib'

# Load the classifier
vectorizer = joblib.load(VECTORIZER_PATH)
classifier = joblib.load(MODEL_PATH)


@app.route('/nlp', methods=['POST'])
def nlp():
    data = request.form.get('data')
    print("request data length: {}".format(len(data)))
    prediction = classifier.predict(vectorizer.transform([data]))
    return jsonify(prediction=prediction[0])
