# Write predictions in csv file.
from statistics import median, mode, mean

import pandas as pd
from mlxtend.plotting import plot_decision_regions
import numpy as np
import seaborn as sns
import seaborn as sn
from matplotlib import pyplot as plt, pyplot
from numpy import hstack, meshgrid, arange
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
#from sklearn.inspection import DecisionBoundaryDisplay


def write_to_csv(IDs, file_name, predictions):
    with open(file_name, 'w') as f:
        f.write("ID,Segmentation\n")
        for i in range(len(predictions)):
            f.write(str(IDs[i]) + ',' + str(predictions[i]) + '\n')


def normalize_data(data):
    scaler = MinMaxScaler()
    scaler_fit = scaler.fit(data)
    data = scaler_fit.transform(data)
    return data


def standardize_data(data):
    # standardization of dependent variables
    data = preprocessing.scale(data)
    return data


def replace_nulls(train_data, test_data):
    train_data['Ever_Married'] = train_data['Ever_Married'].fillna(mode(train_data['Ever_Married']))
    test_data['Ever_Married'] = test_data['Ever_Married'].fillna(mode(test_data['Ever_Married']))
    train_data['Graduated'] = train_data['Graduated'].fillna(mode(train_data['Graduated']))
    test_data['Graduated'] = test_data['Graduated'].fillna(mode(test_data['Graduated']))
    train_data['Work_Experience'] = train_data['Work_Experience'].fillna(median(train_data['Work_Experience']))
    test_data['Work_Experience'] = test_data['Work_Experience'].fillna(median(test_data['Work_Experience']))
    train_data['Family_Size'] = train_data['Family_Size'].fillna(mean(train_data['Family_Size']))
    test_data['Family_Size'] = test_data['Family_Size'].fillna(mean(test_data['Family_Size']))
    train_data['Profession'] = train_data['Profession'].fillna(mode(train_data['Profession']))
    test_data['Profession'] = test_data['Profession'].fillna(mode(test_data['Profession']))
    train_data['Var_1'] = train_data['Var_1'].fillna(mode(train_data['Var_1']))
    test_data['Var_1'] = test_data['Var_1'].fillna(mode(test_data['Var_1']))
    return train_data, test_data



def replace_nulls_with_median(data):
    # standardization of dependent variables
    data['Ever_Married'] = data['Ever_Married'].fillna(median(data['Ever_Married']))
    data['Graduated'] = data['Graduated'].fillna(median(data['Graduated']))
    data['Work_Experience'] = data['Work_Experience'].fillna(median(data['Work_Experience']))
    data['Family_Size'] = data['Family_Size'].fillna(median(data['Family_Size']))
    return data


def replace_nulls_with_mode(data):
    # standardization of dependent variables
    data['Ever_Married'] = data['Ever_Married'].fillna(mode(data['Ever_Married']))
    data['Graduated'] = data['Graduated'].fillna(mode(data['Graduated']))
    data['Work_Experience'] = data['Work_Experience'].fillna(mode(data['Work_Experience']))
    # data['Profession'] = data['Profession'].fillna(mode(data['Profession']))
    data['Family_Size'] = data['Family_Size'].fillna(mode(data['Family_Size']))
    return data


def print_correlation_matrix(data):
    corr_mat = data.corr()
    sn.heatmap(corr_mat, annot=True)
    plt.show()


def convert_segmentation_to_int(data):
    data['Segmentation'] = data['Segmentation'].replace(['A'], 0)
    data['Segmentation'] = data['Segmentation'].replace(['B'], 1)
    data['Segmentation'] = data['Segmentation'].replace(['C'], 2)
    data['Segmentation'] = data['Segmentation'].replace(['D'], 3)
    return data


def convert_segmentation_to_string(data):
    newList = []
    print(data)
    for i in range(len(data)):
        print(data[i])
        if data[i] == 0:
            newList.append('A')
        elif data[i] == 1:
            newList.append('B')
        elif data[i] == 2:
            newList.append('C')
        elif data[i] == 3:
            newList.append('D')
    return newList


def get_model_data(train_data, test_data):
    train_data = train_data.drop(['Var_1_Cat_5'], axis=1)
    test_data = test_data.drop(['Var_1_Cat_5'], axis=1)
    train_data = train_data.drop(['Var_1_Cat_1'], axis=1)
    test_data = test_data.drop(['Var_1_Cat_1'], axis=1)

    # saving test data ids.
    ids = test_data["ID"]

    # dropping the id column
    train_data = train_data.drop(['ID'], axis=1)
    test_data = test_data.drop(['ID'], axis=1)

    # assigning features and targets
    x = train_data.drop(['Segmentation'], axis=1).values
    y = train_data['Segmentation'].values.reshape(-1, 1)
    return x, y, ids, test_data


def stKfoldCrossVal(clf, x, y):
    clf = GradientBoostingClassifier(n_estimators=100, max_features="auto", random_state=23)
    stratifiedkf = StratifiedKFold(n_splits=5)
    score = cross_val_score(clf, x, y, cv=stratifiedkf)
    # v=cross_val_score(clf, x, y, cv=stratifiedkf)
    # print(v)
    print("Cross Validation Scores are {}".format(score))
    print("Average Cross Validation score :{}".format(score.mean()))


def discretization(dataset):
    est = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform')
    new = est.fit_transform(dataset[['Age']])
    dataset['Age'] = new
    return dataset


def decision_boundary(x,y,model):

    pca = PCA(n_components=2)
    X_train2 = pca.fit_transform(x)
    model.fit(X_train2, y)
    plot_decision_regions(X_train2, y[:, 0], clf=model, legend=2)
    plt.show()

def get_features_contributions(train_data,model):
    # get features contribution to the model
    '''RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                        max_depth=None, max_features='auto', max_leaf_nodes=None,
                        min_impurity_decrease=0.0,
                        min_samples_leaf=1, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                        oob_score=False, random_state=None, verbose=0,
                        warm_start=False, )
    '''
    columns = train_data.drop(['Segmentation'], axis=1).columns

    feature_imp = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)
    print(feature_imp)

    # Creating a bar plot
    sns.barplot(x=feature_imp, y=feature_imp.index)
    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.show()


'''def easy_decision_boundary(X, y, model):
    colors = "bry"
    ax = plt.gca()
    from mlxtend.plotting import plot_decision_regions
    DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        cmap=plt.cm.Paired,
        ax=ax,
        response_method="predict",
        xlabel="Age",
        ylabel=["A", "B", "C", "D"],
    )
    # Plot also the training points
    for i, color in zip(model.classes_, colors):
        idx = np.where(y == i)
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            c=color,
            label=["A","B","C","D",][i],
            cmap=plt.cm.Paired,
            edgecolor="black",
            s=20,
        )
    plt.title("Decision surface of multi-class SGD")
    plt.axis("tight")

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    coef = model.coef_
    intercept = model.intercept_

    def plot_hyperplane(c, color):
        def line(x0):
            return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]

        plt.plot([xmin, xmax], [line(xmin), line(xmax)], ls="--", color=color)

    for i, color in zip(model.classes_, colors):
        plot_hyperplane(i, color)
    plt.legend()
    plt.show()'''
