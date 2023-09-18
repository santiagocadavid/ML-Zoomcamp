import pandas as pd
import numpy as np



PATH = r"D:\machine-learning-zoomcamp\raw.githubusercontent.com_alexeygrigorev_datasets_master_housing.csv"


data = pd.read_csv(PATH)


# 1. What's the version of Pandas that you installed?

print(pd.__version__)


# First 5 rows

data_top = data.head()

# 2. How many columns are in the dataset?


data_info = data.info()


#Data columns (total 10 columns):
# #   Column              Non-Null Count  Dtype
#---  ------              --------------  -----
# 0   longitude           20640 non-null  float64
# 1   latitude            20640 non-null  float64
# 2   housing_median_age  20640 non-null  float64
# 3   total_rooms         20640 non-null  float64
# 4   total_bedrooms      20433 non-null  float64
# 5   population          20640 non-null  float64
# 6   households          20640 non-null  float64
# 7   median_income       20640 non-null  float64
# 8   median_house_value  20640 non-null  float64
# 9   ocean_proximity     20640 non-null  object

# Answer: 10

# 3. Which columns in the dataset have missing values?

list_missing_data = data.columns[data.isna().any()].tolist()


# Answer: ['total_bedrooms']

# 4. How many unique values does the ocean_proximity column have?

unique_ocean_proximity = data.ocean_proximity.unique().tolist()

# ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND']
# Answer: 5 values

# 5. What's the average value of the median_house_value for the houses located near the bay?

median_house_value_mean = data.loc[data['ocean_proximity']=='NEAR BAY', 'median_house_value'].mean()

# Answer: 259212.31179039303

# 6. Calculate the average of total_bedrooms column in the dataset.
# Use the fillna method to fill the missing values in total_bedrooms with the mean value from the previous step.
# Now, calculate the average of total_bedrooms again.
# Has it changed?

total_bedrooms_mean = data['total_bedrooms'].mean()
#

#print(total_bedrooms_mean)

#

total_bedrooms_mean = data['total_bedrooms'].fillna(value=total_bedrooms_mean).mean()

#
#print(total_bedrooms_mean)

# Answer: NO

# Question 7

# Select all the options located on islands.
# Select only columns housing_median_age, total_rooms, total_bedrooms.
# Get the underlying NumPy array. Let's call it X.
# Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
# Compute the inverse of XTX.
# Create an array y with values [950, 1300, 800, 1000, 1300].
# Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
# What's the value of the last element of w?

data_island = data[data['ocean_proximity'] == 'ISLAND'][['housing_median_age', 'total_rooms', 'total_bedrooms']]

# X MATRIX
X = np.array(data_island)

# X TRANSPOSED MATRIX
XT = X.T


# X TRANSPOSED MATRIX * X
XTX =  XT @ X

XTX_INV = np.linalg.inv(XTX)

y = [950, 1300, 800, 1000, 1300]

v = XTX_INV @ XT

w = np.dot(v, y)

# Print matrices

#print(X)
#print(XT)
#print(XTX)
#print(XTX_INV)
#print(v)

print(w)

