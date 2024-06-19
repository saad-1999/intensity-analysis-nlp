from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'Invalid input, no text found'}), 400

    text = data['text']
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)
    response = {'sentiment': prediction[0]}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
