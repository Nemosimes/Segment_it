from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PreProcessing.Preprocessing import preprocessing
from helper_functions import write_to_csv, get_model_data

train_data, test_data = preprocessing(0)
x, y, ids, test_data = get_model_data(train_data, test_data)

# print(x)

model = SVC(kernel='linear')
model.fit(x, y)
y_pred = model.predict(test_data)
write_to_csv(ids, '../predictions/predictedFromSVM.csv', y_pred)

# Splitting the dataset into training and test set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Predicting the test set result
model = SVC(kernel='poly')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Acc: ", accuracy_score(y_test, y_pred))
