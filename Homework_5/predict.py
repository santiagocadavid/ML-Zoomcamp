import requests
import pickle
from flask import Flask, request

# Question 4
# Now let's serve this model as a web service
# 
# Install Flask and gunicorn (or waitress, if you're on Windows)
# Write Flask code for serving the model
# Now score this client using requests:


url = 'http://127.0.0.1:9090/predict'

client = {"job": "unknown", "duration": 270, "poutcome": "failure"}

resp = requests.post(url, json=client).json()

print(resp)


# {'credit_probability': 0.13968947052356817}
# Answer: 0.140



