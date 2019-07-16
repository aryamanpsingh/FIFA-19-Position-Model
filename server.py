# Import libraries
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import pickle
app = Flask(__name__)
# Load the model
model = pickle.load(open('model.pkl','rb'))

@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    new_vals = pd.DataFrame(columns=['Index'])
    new_vals = new_vals.append(data['exp'])
    new_vals = new_vals.drop('Index',axis=1)
    print(new_vals)
    prediction = model.predict(np.asarray(data['exp']).reshape((1, -1)))
    # Take the first value of prediction
    output = int(prediction[0])
    return jsonify(output)
if __name__ == '__main__':
    app.run(port=3000, debug=True)