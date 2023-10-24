import re
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text
import xgboost as xgb


PATH = r"D:\machine-learning-zoomcamp\housing.csv"

data_orig = pd.read_csv(PATH)


# First, keep only the records where ocean_proximity is either '<1H OCEAN' or 'INLAND'

# Preparing the dataset
# Preparation:
# 
# Fill missing values with zeros.
# Apply the log transform to median_house_value.
# Do train/validation/test split with 60%/20%/20% distribution.
# Use the train_test_split function and set the random_state parameter to 1.
# Use DictVectorizer(sparse=True) to turn the dataframes into matrices.


# print(data_orig.head())

ocean_proximity_val = ['<1H OCEAN','INLAND']

selected_cols = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", 
               "median_income", "median_house_value", "ocean_proximity"]

df = pd.read_csv(PATH, usecols=selected_cols)
df.total_bedrooms = df.total_bedrooms.fillna(0)

# print(missing_values)

df.total_bedrooms = df.total_bedrooms.fillna(0)


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = np.log1p(df_train.median_house_value.values)
y_val = np.log1p(df_val.median_house_value.values)
y_test = np.log1p(df_test.median_house_value.values)

del df_train['median_house_value']
del df_val['median_house_value']
del df_test['median_house_value']


features = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", 
               "median_income", "ocean_proximity"]

train_dicts = df_train[features].to_dict(orient='records')
val_dicts = df_val[features].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

# Question 1
# Let's train a decision tree regressor to predict the median_house_value variable.
# 
# Train a model with max_depth=1.
# Which feature is used for splitting the data?
# 
# ocean_proximity
# total_rooms
# latitude
# population


dt = DecisionTreeRegressor(max_depth=1)
dt.fit(X_train, y_train)

# print(export_text(dt, feature_names=dv.feature_names_))

# |--- ocean_proximity=INLAND <= 0.50
# |   |--- value: [12.31]
# |--- ocean_proximity=INLAND >  0.50
# |   |--- value: [11.61]


# Answer ocean_proximity=INLAND


# Question 2
# Train a random forest model with these parameters:

# n_estimators=10
# random_state=1
# n_jobs=-1 (optional - to make training faster)

# What's the RMSE of this model on validation?

rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)
msq_error = np.sqrt(mean_squared_error(y_val, y_pred))

# print(msq_error)

#Answer 0.2457383433183843

# Question 3
# Now let's experiment with the n_estimators parameter
# 
# Try different values of this parameter from 10 to 200 with step 10.
# Set random_state to 1.
# Evaluate the model on the validation dataset.
# After which value of n_estimators does RMSE stop improving?

# scores = []

# for n in tqdm(range(10, 201, 10)):
#     rf = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
#     rf.fit(X_train, y_train)
#     
#     y_pred = rf.predict(X_val)
#     score = np.sqrt(mean_squared_error(y_val, y_pred))
#     
#     scores.append((n, score))

# df_scores = pd.DataFrame(scores, columns=['n_estimators', 'rmse'])

# plt.plot(df_scores.n_estimators, df_scores.rmse.round(3))
# plt.show()

# Answer: 50

# Question 4
# Let's select the best max_depth:
# 
# Try different values of max_depth: [10, 15, 20, 25]
# For each of these values,
# try different values of n_estimators from 10 till 200 (with step 10)
# calculate the mean RMSE
# Fix the random seed: random_state=1
# What's the best max_depth, using the mean RMSE?

# scores = []
# 
# for d in tqdm([10, 15, 20, 25]):
#     rf = RandomForestRegressor(n_estimators=0,
#                                max_depth=d,
#                                random_state=1, n_jobs=-1,
#                                warm_start=True)
# 
#     for n in tqdm(range(10, 201, 10)):
#         rf.n_estimators = n
#         rf.fit(X_train, y_train)
# 
#         y_pred = rf.predict(X_val)
#         score = np.sqrt(mean_squared_error(y_val, y_pred))
# 
#         scores.append((d, n, score))
# 
# columns = ['max_depth', 'n_estimators', 'rmse']
# df_scores = pd.DataFrame(scores, columns=columns)
# 
# for d in [10, 15, 20, 25]:
#     df_subset = df_scores[df_scores.max_depth == d]
#     plt.plot(df_subset.n_estimators, df_subset.rmse, label=d)
# 
# plt.legend()
# plt.show()
# Answer 25


# Question 5
# We can extract feature importance information from tree-based models.

# At each step of the decision tree learning algorithm, it finds the best split. When doing it, we can calculate "gain" - the reduction in impurity before and after the split. This gain is quite useful in understanding what are the important features for tree-based models.

# In Scikit-Learn, tree-based models contain this information in the feature_importances_ field.

# For this homework question, we'll find the most important feature:

# Train the model with these parameters:
# n_estimators=10,
# max_depth=20,
# random_state=1,
# n_jobs=-1 (optional)
# Get the feature importance information from this model
# What's the most important feature (among these 4)?


# rf = RandomForestRegressor(n_estimators=10, max_depth=20, 
#                            random_state=1, n_jobs=-1)
# rf.fit(X_train, y_train)

# print(rf.feature_importances_)

# df_importances = pd.DataFrame()
# df_importances['feature'] = dv.feature_names_
# df_importances['importance'] = rf.feature_importances_

# print(df_importances)

# print(df_importances.sort_values(by='importance', ascending=False).head())


# [1.69573183e-02 3.30938997e-02 1.01333971e-01 9.62649876e-02
#  3.62912907e-01 3.00723750e-03 3.10900842e-01 3.56806263e-04
#  4.48661972e-04 4.22762446e-03 3.09180197e-02 1.90412562e-02
#  2.05364687e-02]

#                        feature  importance
# 0                   households    0.016957
# 1           housing_median_age    0.033094
# 2                     latitude    0.101334
# 3                    longitude    0.096265
# 4                median_income    0.362913
# 5    ocean_proximity=<1H OCEAN    0.003007
# 6       ocean_proximity=INLAND    0.310901
# 7       ocean_proximity=ISLAND    0.000357
# 8     ocean_proximity=NEAR BAY    0.000449
# 9   ocean_proximity=NEAR OCEAN    0.004228
# 10                  population    0.030918
# 11              total_bedrooms    0.019041
# 12                 total_rooms    0.020536


#                    feature  importance
# 4           median_income    0.362913
# 6  ocean_proximity=INLAND    0.310901
# 2                latitude    0.101334
# 3               longitude    0.096265
# 1      housing_median_age    0.033094

# Answer: median_income


# Question 6
# Now let's train an XGBoost model! For this question, we'll tune the eta parameter:

# Install XGBoost
# Create DMatrix for train and validation
# Create a watchlist
# Train a model with these parameters for 100 rounds:



xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

# Now change eta from 0.3 to 0.1.

# Which eta leads to the best RMSE score on the validation dataset?

features = dv.feature_names_

regex = re.compile(r"<", re.IGNORECASE)
features = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in features]

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

watchlist = [(dtrain, 'train'), (dval, 'val')]
scores = {}

def parse_xgb_output(output):
    results = []

    for line in output.stdout.strip().split('\n'):
        it_line, train_line, val_line = line.split('\t')

        it = int(it_line.strip('[]'))
        train = float(train_line.split(':')[1])
        val = float(val_line.split(':')[1])

        results.append((it, train, val))

    columns = ['num_iter', 'train_auc', 'val_auc']
    df_results = pd.DataFrame(results, columns=columns)
    return df_results

xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,

    'objective': 'reg:squarederror',
    'nthread': 8,

    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=100,
                  verbose_eval=5, evals=watchlist)

scores['eta=0.3'] = parse_xgb_output(xgb_params)

xgb_params = {
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,

    'objective': 'reg:squarederror',
    'nthread': 8,

    'seed': 1,
    'verbosity': 1,
}


model = xgb.train(xgb_params, dtrain, num_boost_round=100,
                  verbose_eval=5, evals=watchlist)

scores['eta=0.1'] = parse_xgb_output(xgb_params)

plt.plot(scores['eta=0.1'].num_iter, scores['eta=0.1'].val_auc,
        label='0.1')
plt.plot(scores['eta=0.3'].num_iter, scores['eta=0.3'].val_auc,
        label='0.3')
plt.legend()
plt.show()


# Answer: Both gives same