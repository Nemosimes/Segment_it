from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PreProcessing.Preprocessing import preprocessing
from helper_functions import write_to_csv

train_data, test_data = preprocessing(mode=0)
IDs = test_data["ID"]
# data
train_data = train_data.drop(['ID'], axis=1)
test_data = test_data.drop(['ID'], axis=1)
y = train_data['Segmentation'].values.ravel()
x = train_data.drop(['Segmentation'], axis=1).values
# print(x)

model = SVC(kernel='linear')
model.fit(x, y)
y_pred = model.predict(test_data)
write_to_csv(IDs, '../predictions/predictedFromSVM.csv', y_pred)

# Splitting the dataset into training and test set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Predicting the test set result
model = SVC(kernel='poly')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Acc: ", accuracy_score(y_test, y_pred))
