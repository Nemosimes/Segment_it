# importing utility modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from helper_functions import write_to_csv
from sklearn.metrics import accuracy_score
from PreProcessing.Preprocessing import preprocessing
# importing machine learning models for prediction
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


# importing voting classifier
from sklearn.ensemble import VotingClassifier

train_data, test_data = preprocessing(mode=1)
train_data = train_data.drop(['Var_1_Cat_5'], axis=1)
test_data = test_data.drop(['Var_1_Cat_5'], axis=1)
train_data = train_data.drop(['Var_1_Cat_1'], axis=1)
test_data = test_data.drop(['Var_1_Cat_1'], axis=1)

IDs = test_data["ID"]
# data
train_data = train_data.drop(['ID'], axis=1)
test_data = test_data.drop(['ID'], axis=1)
y = train_data['Segmentation'].values.ravel()
x = train_data.drop(['Segmentation'], axis=1).values
# print(x)


# initializing all the model objects with default parameters

model_1 = DecisionTreeClassifier(criterion='entropy', random_state=0)
model_2 = SVC(C= 1000, gamma= 0.001, kernel='rbf',probability = True)
model_3 =GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

estimator = [('Dt', model_1), ('SVC', model_2), ('GradientBoosting', model_3)]
# Making the final model using voting classifier
final_model = VotingClassifier(estimators=estimator, voting='hard')

# training all the model on the train dataset
final_model.fit(x, y)
pred_final = final_model.predict(test_data)
write_to_csv(IDs, '../predictions/predictedFromEnsmbleVoting.csv', pred_final)




# Splitting the dataset into training and test set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Predicting the test set result

model = VotingClassifier(estimators=estimator, voting='hard')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


from sklearn.metrics import classification_report, confusion_matrix
# print classification report
print(classification_report(y_test, y_pred))

print("Acc: ", accuracy_score(y_test, y_pred))




