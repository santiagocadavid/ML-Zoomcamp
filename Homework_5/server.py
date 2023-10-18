import requests
import pickle
from flask import Flask, request, jsonify

# Question 4
# Now let's serve this model as a web service
# 
# Install Flask and gunicorn (or waitress, if you're on Windows)
# Write Flask code for serving the model
# Now score this client using requests:



path_model = r"D:\machine-learning-zoomcamp\model1.bin"
path_dict =  r"D:\machine-learning-zoomcamp\dv.bin"

with open(path_model, 'rb') as f_in_m:
    model = pickle.load(f_in_m)


with open(path_dict, 'rb') as f_in_d:
    dv = pickle.load(f_in_d)

app = Flask('credit')

@app.route('/predict', methods = ['POST'])
def predict():
    
    Score = request.get_json()
    X = dv.transform([Score])
    y_pred = model.predict_proba(X)[0, 1]

    result = {
        'credit_probability':y_pred
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9090)







