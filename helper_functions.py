# Write predictions in csv file.
from statistics import median, mode
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn import preprocessing
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
    data['Family_Size'] = data['Family_Size'].fillna(mode(data['Family_Size']))
    return data


def print_correlation_matrix(data):
    corr_mat = data.corr()
    sn.heatmap(corr_mat, annot=True)
    plt.show()


def convert_segmentation_to_int(data):
    data['Segmentation'] = data['Segmentation'].replace(['A'], 1)
    data['Segmentation'] = data['Segmentation'].replace(['B'], 2)
    data['Segmentation'] = data['Segmentation'].replace(['C'], 3)
    data['Segmentation'] = data['Segmentation'].replace(['D'], 4)
    return data


def convert_segmentation_to_string(data):
    data['Segmentation'] = data['Segmentation'].replace([1], 'A')
    data['Segmentation'] = data['Segmentation'].replace([2], 'B')
    data['Segmentation'] = data['Segmentation'].replace([3], 'C')
    data['Segmentation'] = data['Segmentation'].replace([4], 'D')
    return data


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
