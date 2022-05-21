from sklearn.model_selection import train_test_split  # for splitting the data into train and test samples
from sklearn.metrics import classification_report  # for model evaluation metrics
from xgboost import XGBClassifier  # for extreme gradient boosting model

from PreProcessing.Preprocessing import preprocessing
from helper_functions import get_model_data, convert_segmentation_to_int, convert_segmentation_to_string, write_to_csv

train_data, test_data = preprocessing(1)
print(train_data.columns)
print(test_data.columns)
train_data = convert_segmentation_to_int(train_data)
print(train_data.columns)
x, y, ids, test_data = get_model_data(train_data, test_data)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)
print(x_train)
print(x_test)
print(y_train)

model = XGBClassifier(use_label_encoder=False,
                      booster='dart',  # boosting algorithm to use, default gbtree, othera: gblinear, dart
                      n_estimators=500,  # number of trees, default = 100
                      eta=0.3,  # this is learning rate, default = 0.3
                      max_depth=6,  # maximum depth of the tree, default = 6
                      gamma=4,  # used for pruning, if gain < gamma the branch will be pruned, default = 0
                      reg_lambda=1,  # regularization parameter, defautl = 1
                      # min_child_weight=0 # this refers to Cover which is also responsible for pruning if not set to 0
                      )

# Fit the model
clf = model.fit(x_train, y_train)

##### Step 3
# Predict class labels on training data
pred_labels_tr = model.predict(x_train)
# Predict class labels on a test data
pred_labels_te = model.predict(x_test)

# Basic info about the model
print('*************** Tree Summary ***************')
print('No. of classes: ', clf.n_classes_)
print('Classes: ', clf.classes_)
print('No. of features: ', clf.n_features_in_)
print('No. of Estimators: ', clf.n_estimators)
print('--------------------------------------------------------')
print("")

print('*************** Evaluation on Test Data ***************')
score_te = model.score(x_test, y_test)
print('Accuracy Score: ', score_te)
# Look at classification report to evaluate the model
print(classification_report(y_test, pred_labels_te))
print('--------------------------------------------------------')
print("")

print('*************** Evaluation on Training Data ***************')
score_tr = model.score(x_train, y_train)
print('Accuracy Score: ', score_tr)
# Look at classification report to evaluate the model
print(classification_report(y_train, pred_labels_tr))
print('--------------------------------------------------------')

predictions = model.predict(test_data)
print(predictions)
predictions = convert_segmentation_to_string(predictions)
print(predictions)
write_to_csv(ids, '/Users/pierreehab/Desktop/Segment_it/predictions/predictedFromXGBoosting.csv', predictions)