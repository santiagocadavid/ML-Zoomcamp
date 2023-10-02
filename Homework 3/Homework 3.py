import random
import pandas as pd
import numpy as np
from statistics import mode
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split


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

#Question 1
#What is the most frequent observation (mode) for the column transmission_type?

mode_transmission = mode(data['transmission_type'])

#print(mode_transmission)

#ANSWER: AUTOMATIC


#Question 2

#Create the correlation matrix for the numerical features of your dataset. In a correlation matrix, you compute the correlation coefficient between every pair of features in the dataset.
#
#What are the two features that have the biggest correlation in this dataset?

corr_matrix = data.corr()

#print(corr_matrix)

#                        year  engine_hp  engine_cylinders  highway_mpg  city_mpg     price
#  year              1.000000   0.338714         -0.040708     0.258240  0.198171  0.227590
#  engine_hp         0.338714   1.000000          0.774851    -0.415707 -0.424918  0.650095
#  engine_cylinders -0.040708   0.774851          1.000000    -0.614541 -0.587306  0.526274
#  highway_mpg       0.258240  -0.415707         -0.614541     1.000000  0.886829 -0.160043
#  city_mpg          0.198171  -0.424918         -0.587306     0.886829  1.000000 -0.157676
#  price             0.227590   0.650095          0.526274    -0.160043 -0.157676  1.000000

plt.figure(figsize=(15,15))  
sns.heatmap(data.corr(),annot=True,linewidths=.8, cmap="Reds")
plt.title('Heatmap showing correlations between numerical data')
plt.show()

# ANSWER The two features with biggest correlation are highway_mpg and city_mpg
# with 0.886829

# Now we need to turn the price variable from numeric into a binary format.
# Let's create a variable above_average which is 1 if the price is above its mean value and 0 otherwise.


data_class = data.copy()

mean = data_class['price'].mean()

data_class['above_average'] = np.where(data_class['price']>=mean,1,0)


data_class = data_class.drop('price', axis=1)


df_train_full, df_test = train_test_split(data_class, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.above_average.values
y_val = df_val.above_average.values
y_test = df_test.above_average.values


# Question 3
# Calculate the mutual information score between above_average and other categorical variables in our dataset. Use the training set only.
# Round the scores to 2 decimals using round(score, 2).

cat = ['make','model','transmission_type','vehicle_style']


def calculate_mi(series):

    return mutual_info_score(series, df_train.above_average)


df_mi = df_train[cat].apply(calculate_mi)
df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')

#print(df_mi)

#                          MI
# model              0.462344
# make               0.239769
# vehicle_style      0.084143
# transmission_type  0.020958

# ANSWER: transmission_type


# Question 4

# Now let's train a logistic regression.
# Remember that we have several categorical variables in the dataset. Include them using one-hot encoding.
# Fit the model on the training dataset.
# To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters:
# model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)
# Calculate the accuracy on the validation dataset and round it to 2 decimal digits.

num = ["year", "engine_hp", "engine_cylinders", "highway_mpg", "city_mpg"]


train_dict = df_train[cat + num].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
dv.fit(train_dict)
X_train = dv.transform(train_dict)

model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

val_dict = df_val[cat + num].to_dict(orient='records')
X_val = dv.transform(val_dict)
y_pred = model.predict(X_val)

accuracy = np.round(accuracy_score(y_val, y_pred),2)

# print(accuracy)

#ANSWER 0.95

# Question 5

# Let's find the least useful feature using the feature elimination technique.
# Train a model with all these features (using the same parameters as in Q4).
# Now exclude each feature from this set and train a model without it. Record the accuracy for each model.
# For each feature, calculate the difference between the original accuracy and the accuracy without the feature.

features = cat + num


orig_score = accuracy

for i in features:
    subset = features.copy()
    subset.remove(i)
    
    train_dict = df_train[subset].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    dv.fit(train_dict)

    X_train = dv.transform(train_dict)

    model = LogisticRegression(solver='liblinear', C=10, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    val_dict = df_val[subset].to_dict(orient='records')
    X_val = dv.transform(val_dict)

    y_pred = model.predict(X_val)

    score = accuracy_score(y_val, y_pred)
    print(i, orig_score - score, score)


# make 0.0032941670163658676 0.9467058329836341
# model 0.025954678976080503 0.9240453210239195
# transmission_type 0.004972723457826178 0.9450272765421738
# vehicle_style 0.01756189676877884 0.9324381032312211
# year 0.0020352496852705793 0.9479647503147294
# engine_hp 0.02217792698279475 0.9278220730172052
# engine_cylinders 0.004133445237096023 0.9458665547629039
# highway_mpg 0.0032941670163658676 0.9467058329836341
# city_mpg 0.01756189676877884 0.9324381032312211

# ANSWER: year 0.0020352496852705793

# Question 6

# For this question, we'll see how to use a linear regression model from Scikit-Learn.
# We'll need to use the original column price. Apply the logarithmic transformation to this column.
# Fit the Ridge regression model on the training data with a solver 'sag'. Set the seed to 42.
# This model also has a parameter alpha. Let's try the following values: [0, 0.01, 0.1, 1, 10].
# Round your RMSE scores to 3 decimal digits.


data['price']=np.log1p(data['price'])

df_train_full, df_test = train_test_split(data, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


y_train = df_train.price.values
y_val = df_val.price.values
y_test = df_test.price.values


del df_train['price']
del df_val['price']
del df_test['price']


train_dict = df_train[cat + num].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)

val_dict = df_val[cat + num].to_dict(orient='records')
X_val = dv.transform(val_dict)


for a in [0, 0.01, 0.1, 1, 10]:
    model = Ridge(alpha=a, solver="sag", random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    
    score = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(a, round(score, 4))


# 0 0.4868
# 0.01 0.4868
# 0.1 0.4868
# 1 0.4868
# 10 0.4868

#ANSWER 0

