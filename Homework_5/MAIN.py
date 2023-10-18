import pickle


#Question 1
#Install Pipenv
#What's the version of pipenv you installed?
#Use --version to find out


# Answer: pipenv, version 2023.10.3


# Question 2

# What's the first hash for scikit-learn you get in Pipfile.lock?

# 0e0fec5cb0e411bbb2c1c4f81b061609272a25d0c1f780d06dd30aff281bed02


# Question 3
# Let's use these models!
# 
# Write a script for loading these models with pickle
# Score this client:
# {"job": "retired", "duration": 445, "poutcome": "success"}


path_model = r"D:\machine-learning-zoomcamp\model1.bin"
path_dict =  r"D:\machine-learning-zoomcamp\dv.bin"

with open(path_model, 'rb') as f_in_m:
    model = pickle.load(f_in_m)


with open(path_dict, 'rb') as f_in_d:
    dv = pickle.load(f_in_d)


Score = {
        "job": "retired", 
         "duration": 445, 
         "poutcome": "success"
         }

X = dv.transform([Score])
y_pred = model.predict_proba(X)[0, 1]
 
print("Input: ", Score)
print("Credit probability: ", y_pred)

# Input:  {'job': 'retired', 'duration': 445, 'poutcome': 'success'}
# Credit probability:  0.9019309332297606


# Question 4
# {'credit_probability': 0.13968947052356817}
# Answer 0.140


# Question 5
# Download the base image svizor/zoomcamp-model:3.10.12-slim. You can easily make it by using docker pull command.

# Answer: 147MB


# Question 6
# Let's run your docker container!

# After running it, score this client once again:

# Answer: 0.968
