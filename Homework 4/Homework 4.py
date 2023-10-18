from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split



PATH = r"D:\machine-learning-zoomcamp\raw.githubusercontent.com_alexeygrigorev_mlbookcamp-code_master_chapter-02-car-price_data.csv"


data_orig = pd.read_csv(PATH)

#Select columns

data = data_orig[['Make','Model','Year','Engine HP','Engine Cylinders','Transmission Type','Vehicle Style','highway MPG','city mpg','MSRP']]


#transform names of columns

data.columns = data.columns.str.replace(' ', '_').str.lower()


#Find missing values

missingvalues = data.columns[data.isna().any()].to_list()

#print(missingvalues)

data = data.fillna(0)

data.rename(columns={'msrp':'price'}, inplace=True)

print(data.head())


data_class = data.copy()

mean = data_class['price'].mean()

data_class['above_average'] = np.where(data_class['price']>=mean,1,0)


data_class = data_class.drop('price', axis=1)


df_full_train, df_test = train_test_split(data_class, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.above_average.values
y_val = df_val.above_average.values
y_test = df_test.above_average.values

# Question 1: ROC AUC feature importance

# ROC AUC could also be used to evaluate feature importance of numerical variables.
# 
# Let's do that
# 
# For each numerical variable, use it as score and compute AUC with the above_average variable
# Use the training dataset for that
# If your AUC is < 0.5, invert this variable by putting "-" in front
# 
# (e.g. -df_train['engine_hp'])
# 
# AUC can go below 0.5 if the variable is negatively correlated with the target varialble. You can change the direction of the correlation by negating this variable - then negative correlation becomes positive.
# 
# Which numerical variable (among the following 4) has the highest AUC?

numerical = ["year", "engine_hp", "engine_cylinders", "highway_mpg", "city_mpg"]

categorical = ['make','model','transmission_type','vehicle_style']

for c in numerical:
    auc = roc_auc_score(y_train, df_train[c])
    if auc < 0.5:
        auc = roc_auc_score(y_train, -df_train[c])
    print('%9s, %.3f' % (c, auc))


# year, 0.688
# engine_hp, 0.917
# engine_cylinders, 0.766
# highway_mpg, 0.633
# city_mpg, 0.673

# Answer: engine_hp

# Question 2: Training the model
# Apply one-hot-encoding using DictVectorizer and train the logistic regression with these parameters:

# LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
# What's the AUC of this model on the validation dataset? (round to 3 digits)

columns = categorical + numerical

train_dicts = df_train[columns].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

val_dicts = df_val[columns].to_dict(orient='records')
X_val = dv.transform(val_dicts)

y_pred = model.predict_proba(X_val)[:, 1]

print(roc_auc_score(y_val, y_pred))

# 0.9790336815928319

# Answer 0.979

# Question 3: Precision and Recall
# Now let's compute precision and recall for our model.
# 
# Evaluate the model on all thresholds from 0.0 to 1.0 with step 0.01
# For each threshold, compute precision and recall
# Plot them
# At which threshold precision and recall curves intersect?



def confusion_matrix_dataframe(y_val, y_pred):
    scores = []

    thresholds = np.linspace(0, 1, 101)

    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()

        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        scores.append((t, tp, fp, fn, tn))

    columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
    df_scores = pd.DataFrame(scores, columns=columns)
    
    return df_scores

df_scores = confusion_matrix_dataframe(y_val, y_pred)
df_scores[::10]

df_scores['p'] = df_scores.tp / (df_scores.tp + df_scores.fp)
df_scores['r'] = df_scores.tp / (df_scores.tp + df_scores.fn)

#plt.plot(df_scores.threshold, df_scores.p, label='precision')
#plt.plot(df_scores.threshold, df_scores.r, label='recall')
#
#plt.legend()
#plt.show()

# Answer ~=0.478


# Question 4: F1 score

# Precision and recall are conflicting - when one grows, the other goes down. That's why they are often combined into the F1 score - a metrics that takes into account both
# 
# This is the formula for computing F1:
# 
#  
# 
# Where 
 # is precision and 
 # is recall.
# 
# Let's compute F1 for all thresholds from 0.0 to 1.0 with increment 0.01
# 
# At which threshold F1 is maximal?

df_scores['f1'] = 2 * df_scores.p * df_scores.r / (df_scores.p + df_scores.r)

# plt.plot(df_scores.threshold, df_scores.f1)
# plt.xticks(np.linspace(0, 1, 11))
# plt.show()

#Answer ~=0.52

# Question 5: 5-Fold CV
# Use the KFold class from Scikit-Learn to evaluate our model on 5 different folds:
# 
# KFold(n_splits=5, shuffle=True, random_state=1)
# Iterate over different folds of df_full_train
# Split the data into train and validation
# Train the model on train with these parameters: LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
# Use AUC to evaluate the model on validation
# How large is standard devidation of the scores across different folds?


def train(df_train, y_train, C=1.0):
    dicts = df_train[columns].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', C=C)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[columns].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


scores = []

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.above_average
    y_val = df_val.above_average

    dv, model = train(df_train, y_train, C=1.0)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))

# 0.979 +- 0.003
#Answer: 0.003

# Question 6: Hyperparemeter Tuning
# Now let's use 5-Fold cross-validation to find the best parameter C
# 
# Iterate over the following C values: [0.01, 0.1, 0.5, 10]
# Initialize KFold with the same parameters as previously
# Use these parametes for the model: LogisticRegression(solver='liblinear', C=C, max_iter=1000)
# Compute the mean score as well as the std (round the mean and std to 3 decimal digits)
# Which C leads to the best mean score?


kfold = KFold(n_splits=5, shuffle=True, random_state=1)

for C in [0.01, 0.1, 1, 10]:
    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.above_average
        y_val = df_val.above_average

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%4s, %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# C=0.01, 0.952 +- 0.002
# C= 0.1, 0.972 +- 0.002
# C=   1, 0.979 +- 0.003
# C=  10, 0.983 +- 0.003

# Answer: 10

