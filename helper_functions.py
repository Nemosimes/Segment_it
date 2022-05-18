# Write predictions in csv file.
from statistics import median, mode, mean
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


def write_to_csv(IDs,file_name, predictions):
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
