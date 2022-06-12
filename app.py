import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, app, jsonify , url_for, render_template

app = Flask(__name__)
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)

    output = model.predict(final_features)[0]
    print(output)
    # output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="Price is  {}".format(output))


if __name__ == "__main__":
    app.run(debug=True)
