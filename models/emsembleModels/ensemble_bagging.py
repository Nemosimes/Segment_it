from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingRegressor, GradientBoostingClassifier
from PreProcessing.Preprocessing import preprocessing
from helper_functions import write_to_csv, get_model_data

train_data, test_data = preprocessing(1)
x, y, ids, test_data = get_model_data(train_data, test_data)


# Splitting between train data into training and validation dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

# initializing the bagging model using XGboost as base model with default parameters
model = BaggingRegressor(base_estimator=GradientBoostingClassifier(n_estimators=100, max_features="auto", random_state=23))

# training model
model.fit(x_train, y_train)

# predicting the output on the test dataset
pred = model.predict(x_test)

# printing the root mean squared error between real value and predicted value
print(mean_squared_error(y_test, pred))

model = BaggingRegressor(base_estimator=GradientBoostingClassifier(n_estimators=100, max_features="auto", random_state=23))

# training model
model.fit(x, y)

# predicting the output on the test dataset
pred = model.predict(test_data)

write_to_csv(ids, '../predictions/predictedFromBagging.csv', pred)
