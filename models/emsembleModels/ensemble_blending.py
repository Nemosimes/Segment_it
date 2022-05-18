# fit all models on the training set and predict on hold out set
from numpy import hstack
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from PreProcessing.Preprocessing import preprocessing
from helper_functions import write_to_csv, get_model_data


def get_models():
    models = list()
    models.append(('lr', LogisticRegression()))
    models.append(('knn', KNeighborsClassifier()))
    models.append(('svm', SVC(probability=True)))
    models.append(('bayes', GaussianNB()))
    models.append(('rfc', RandomForestClassifier(n_estimators=100, max_features="auto", random_state=23)))
    models.append(('dtc', DecisionTreeClassifier(criterion='entropy', random_state=23)))
    models.append(('gbc', GradientBoostingClassifier(n_estimators=100, max_features="auto", random_state=23)))
    return models


def fit_ensemble(models, X_train, X_val, y_train, y_val):
    # fit all models on the training set and predict on hold out set
    meta_X = list()
    for name, model in models:
        # fit in training set
        model.fit(X_train, y_train)
        # predict on hold out set
        yhat = model.predict(X_val)
        # reshape predictions into a matrix with one column
        yhat = yhat.reshape(len(yhat), 1)
        # store predictions as input for blending
        meta_X.append(yhat)
    # create 2d array from predictions, each set is an input feature
    meta_X = hstack(meta_X)
    # define blending model
    blender = LogisticRegression()
    # fit on predictions from base models
    blender.fit(meta_X, y_val)
    return blender


def predict_ensemble(models, blender, X_test):
    # make predictions with base models
    meta_X = list()
    for name, model in models:
        # predict with base model
        yhat = model.predict(X_test)
        # reshape predictions into a matrix with one column
        yhat = yhat.reshape(len(yhat), 1)
        # store prediction
        meta_X.append(yhat)
    # create 2d array from predictions, each set is an input feature
    meta_X = hstack(meta_X)
    # predict
    return blender.predict(meta_X)


train_data, test_data = preprocessing(1)
x, y, ids, test_data = get_model_data(train_data, test_data)

X_train_full, X_test, y_train_full, y_test = train_test_split(x, y, test_size=0.5, random_state=23)
# split training set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.33, random_state=23)

models = get_models()
# train the blending ensemble
blender = fit_ensemble(models, X_train, X_val, y_train, y_val)
# make predictions on test set
yhat = predict_ensemble(models, blender, X_test)

score = accuracy_score(y_test, yhat)
print('Blending Accuracy: %.3f' % score)
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.33, random_state=23)
models = get_models()
# train the blending ensemble
blender = fit_ensemble(models, X_train, X_val, y_train, y_val)
# make predictions on test set
yhat = predict_ensemble(models, blender, test_data)

write_to_csv(ids, '../predictions/predictedFromBlending.csv', yhat)

