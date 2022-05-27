# Write predictions in csv file.
from statistics import median, mode, mean
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler


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

def replace_nulls(train_data,test_data):
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
    return train_data,test_data

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
    #data['Profession'] = data['Profession'].fillna(mode(data['Profession']))
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
    print (data)
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


def stKfoldCrossVal(clf,x,y):
    clf = GradientBoostingClassifier(n_estimators=100, max_features="auto", random_state=23)
    stratifiedkf = StratifiedKFold(n_splits=5)
    score = cross_val_score(clf, x, y, cv=stratifiedkf)
    # v=cross_val_score(clf, x, y, cv=stratifiedkf)
    # print(v)
    print("Cross Validation Scores are {}".format(score))
    print("Average Cross Validation score :{}".format(score.mean()))
