import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from numpy import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from scipy.stats import ttest_rel

def onehot_encode(dataset, column_dict):
    dataset = dataset.copy()
    for column, prefix in column_dict.items():
        dummies = pd.get_dummies(dataset[column], prefix=prefix)
        dataset = pd.concat([dataset, dummies], axis=1)
        dataset = dataset.drop(column, axis=1)
        # print(dummies)
    return dataset

 
    #Find the unique values in the column “feature_name”
    values = data[feature_name].unique()

    entropy=0
    for i in values:
        a = data[data[feature_name] == i]
        prob = (a.shape[0] / data.shape[0])
        entropy += prob*calc_entropy(a[target_name])

    return entropy
    
def pretreatment(dataset, no_input, seed):
    dataset = dataset.copy()
    # Drop the URL column, because it is only the id for the URL, no real significance
    dataset = dataset.drop('URL', axis=1)
        
    # # ---------------------------------Date transform version 1--------------------------------
    # # Extract datetime features
    for column in ['WHOIS_REGDATE', 'WHOIS_UPDATED_DATE']:
        dataset[column] = pd.to_datetime(dataset[column], errors='coerce')
    
    dataset['REG_YEAR'] = dataset['WHOIS_REGDATE'].apply(lambda x: x.year)
    dataset['REG_MONTH'] = dataset['WHOIS_REGDATE'].apply(lambda x: x.month)
    dataset['REG_DAY'] = dataset['WHOIS_REGDATE'].apply(lambda x: x.day)
    dataset['REG_HOUR'] = dataset['WHOIS_REGDATE'].apply(lambda x: x.hour)
    dataset['REG_MINUTE'] = dataset['WHOIS_REGDATE'].apply(lambda x: x.minute)
    
    dataset['UPD_YEAR'] = dataset['WHOIS_UPDATED_DATE'].apply(lambda x: x.year)
    dataset['UPD_MONTH'] = dataset['WHOIS_UPDATED_DATE'].apply(lambda x: x.month)
    dataset['UPD_DAY'] = dataset['WHOIS_UPDATED_DATE'].apply(lambda x: x.day)
    dataset['UPD_HOUR'] = dataset['WHOIS_UPDATED_DATE'].apply(lambda x: x.hour)
    dataset['UPD_MINUTE'] = dataset['WHOIS_UPDATED_DATE'].apply(lambda x: x.minute)
    
    
    # ---------------------------------Date transform version 2--------------------------------
    # for column in ['WHOIS_REGDATE', 'WHOIS_UPDATED_DATE']:
    #     dataset[column] = pd.to_datetime(dataset[column], errors='coerce')
    # dataset['REG_YEAR'] = dataset['WHOIS_REGDATE'].apply(lambda x: x.year)
    # dataset['UPD_YEAR'] = dataset['WHOIS_UPDATED_DATE'].apply(lambda x: x.year)
    # dataset['REG_MONTH'] = dataset['WHOIS_REGDATE'].apply(lambda x: x.month)
    # dataset['UPD_MONTH'] = dataset['WHOIS_UPDATED_DATE'].apply(lambda x: x.month)
    # dataset['REGINFO'] = dataset['WHOIS_REGDATE'].apply(lambda x: 1 if x != 'None' else 0)
    # dataset['UPDINFO'] = dataset['WHOIS_UPDATED_DATE'].apply(lambda x: 1 if x != 'None' else 0)

    # dataset['NO_UPD'] = dataset['UPDINFO'].copy()
    # for i in range(len(dataset['REGINFO'])):
    #     if dataset['REGINFO'].iloc[i] == 1 and dataset['UPDINFO'].iloc[i] == 0:
    #         dataset['NO_UPD'].iloc[i] = 1
    #     else:
    #         dataset['NO_UPD'].iloc[i] = 0
        
    dataset = dataset.drop(['WHOIS_REGDATE', 'WHOIS_UPDATED_DATE'], axis=1)

    # One-hot encode categorical features
    for column in ['CHARSET', 'SERVER', 'WHOIS_COUNTRY', 'WHOIS_STATEPRO']:
        dataset[column] = dataset[column].apply(lambda x: x.lower() if str(x) != 'nan' else x)
    
    dataset = onehot_encode(
        dataset,
        column_dict={
            'CHARSET': 'CH',
            'SERVER': 'SV',
            'WHOIS_COUNTRY': 'WC',
            'WHOIS_STATEPRO': 'WS'
        }
    )
    
    # ---------------------------------Fill missing values version 1--------------------------------
    # missing_value_columns = dataset.columns[dataset.isna().sum() > 0]
    # for column in missing_value_columns:
    #     dataset[column] = dataset[column].fillna(dataset[column].mean())

    # ---------------------------------Fill missing values version 2--------------------------------
    missing_value_columns = dataset.columns[dataset.isna().sum() > 0]
    date_miss = missing_value_columns[2::]
    other_features = missing_value_columns[0:2]
    for column in other_features:
        dataset[column] = dataset[column].fillna(dataset[column].mean())
    for column in date_miss:
        dataset[column] = dataset[column].fillna(0)


    # ---------------------------------Normalization-----------------------------------
    y = dataset['Type'].copy()
    X = dataset.drop('Type', axis=1).copy()
    single_value_columns = X.columns[[len(X[column].unique()) == 1 for column in X.columns]]
    X = X.drop(single_value_columns, axis=1)


    # ---------------------------------Find the valuable Features-----------------------------------
    test = SelectKBest(score_func=chi2, k='all')
    fit = test.fit(MinMaxScaler().fit_transform(X), y) 
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(column for column in X)
    featureScore = pd.concat([dfcolumns, dfscores], axis=1)
    featureScore.columns = ['Features','Score']
    valuable_feature = featureScore.nlargest(no_input,'Score')['Features']
    X = X[valuable_feature]
    
    # ---------------------------------Split dataset into X and y-----------------------------------
    scaler = StandardScaler()
    scaler.fit(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=seed)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    return X_train, X_test, y_train, y_test, X, y

def evaluate_logistic_regression(X_test, y_test, X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    poly = PolynomialFeatures(degree = 2)
    X_poly = poly.fit_transform(X_train)
    model.fit(X_poly, y_train)
    
    X_poly_test = poly.fit_transform(X_test)
    model_acc = model.score(X_poly_test, y_test)
    print("Test Accuracy: {:.2f}%".format(model_acc * 100))
    
    y_true = np.array(y_test)
    y_pred = model.predict(X_poly_test)
    
    cm = confusion_matrix(y_true, y_pred)
    clr = classification_report(y_true, y_pred, target_names=["BENIGN", "MALIGNANT"])
    
    print("Logistic regression Classification Report:\n----------------------\n", clr)

    return y_pred.tolist()

def evaluate_random_forest(X_test, y_test, X_train, y_train):
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train,y_train)

    model_acc = model.score(X_test, y_test)
    print("Test Accuracy: {:.2f}%".format(model_acc * 100))
    
    y_true = np.array(y_test)
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_true, y_pred)
    clr = classification_report(y_true, y_pred, target_names=["BENIGN", "MALIGNANT"])
    
    print("Random Forest Classification Report:\n----------------------\n", clr)
    return y_pred.tolist()

def evaluate_KNN(X_test, y_test, X_train, y_train):
    # modelskNN=[]
    # modelskNN.append(('1', KNeighborsClassifier(n_neighbors=1)))
    # modelskNN.append(('3', KNeighborsClassifier(n_neighbors=3)))
    # modelskNN.append(('5', KNeighborsClassifier(n_neighbors=5)))

    # resultskNN = []
    # nameskNN=[]
    # kfold = StratifiedKFold(n_splits=5)
    # for name, model in modelskNN:
    #     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='f1')
    #     resultskNN.append(cv_results)
    #     nameskNN.append(name)
    #     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    model= KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train,y_train)

    model_acc = model.score(X_test, y_test)
    print("Test Accuracy: {:.2f}%".format(model_acc * 100))
    
    y_true = np.array(y_test)
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_true, y_pred)
    clr = classification_report(y_true, y_pred, target_names=["BENIGN", "MALIGNANT"])
    
    print("KNN Classification Report:\n----------------------\n", clr)
    return y_pred.tolist()

def evaluate_neural(X_test, y_test, X_train, y_train, no_input, layer_1, layer_2, layer_3):
    model = Sequential()
    model.add(Dense(layer_1, input_dim=no_input, activation='relu'))
    model.add(Dense(layer_2, activation='relu'))
    model.add(Dense(layer_3, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0017), metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=10, batch_size=10)
    y_pred = model.predict(X_test)>0.5

    y_true = np.array(y_test)
    cm = confusion_matrix(y_true, y_pred)
    clr = classification_report(y_true, y_pred, target_names=["BENIGN", "MALIGNANT"])
    
    print("Neural Network Classification Report:\n----------------------\n", clr)
    return metrics.f1_score(y_test, y_pred, average=None), y_pred.tolist()

def find_no_input_feature():
    step = 3
    output_score = 0
    res = 0

    for no_input in range(133,149,step):    
        score = []
        for i in range(5):    
            X_train, X_test, y_train, y_test, X, y = pretreatment(dataset, no_input)
            score.append(evaluate_neural(X_test, y_test, X_train, y_train, no_input, 150, 100, 50)[0])

        if  output_score < mean(score):
            output_score = mean(score)
            res = no_input

        print('--------------------------------------')
        print('Progress: ', no_input, '/464')
        print('score = ', output_score)
        print("Output_no_input = ", res)


def find_no_nodes():
    step = 20
    output_score = 0
    res = []
    open("record.txt", "w")
    with open("record.txt", "a") as f:
        for i in range(10,151,step):    
            for j in range(10,151,step):
                for k in range(10,151,step):
                    score = []
                    for p in range(5):    
                        X_train, X_test, y_train, y_test, X, y = pretreatment(dataset, no_input, p)
                        score.append(evaluate_neural(X_test, y_test, X_train, y_train, no_input, i, j, k)[0])
                        record = str(mean(score))+"\n"
                        f.write(record)
                        if  output_score < mean(score):
                            output_score = mean(score)
                            res = [i, j, k]

                    print('--------------------------------------')
                    print('Progress: ', i, '/150', j, '/150', k, '/150')
                    print('score = ', output_score)
                    print("Output_no_nodes = ", res)   

def vote(rd_res, neural_res, y_test):
    res = []
    neural_int_res = []
    for i in range(len(rd_res)):
        if neural_res[i] == [False]:
            neural_int_res.append(0)
        else:
            neural_int_res.append(1)

    for i in range(len(rd_res)):
        if rd_res[i] or neural_int_res[i] == 1:
            res.append(1)
        else:
            res.append(0)
    y_true = np.array(y_test)
    cm = confusion_matrix(y_true, res)
    clr = classification_report(y_true, res, target_names=["BENIGN", "MALIGNANT"])
    
    print("Final Classification Report:\n----------------------\n", clr, "\n")
    
def display_label(y):
    y_1 = []
    y_0 = []
    for i in y:
        if i == 0:
            y_0.append(i)
        else:
            y_1.append(i)
    plt.hist(y_0,range=(0,1))
    plt.hist(y_1,range=(0,1))
    plt.title("Histogram of Benign and Malicious Website")
    plt.legend(["Benign", "Malicious"])
    plt.show()


names = ['URL','URL_LENGTH', 'NUMBER_SPECIAL_CHARACTERS', 'CHARSET', 'SERVER','CONTENT_LENGTH','WHOIS_COUNTRY', 'WHOIS_STATEPRO', 'WHOIS_REGDATE', 'WHOIS_UPDATED_DATE', 'TCP_CONVERSATION_EXCHANGE','DIST_REMOTE_TCP_PORT', 'REMOTE_IPS', 'APP_BYTES', 'SOURCE_APP_PACKETS', 'REMOTE_APP_PACKETS','SOURCE_APP_BYTES','REMOTE_APP_BYTES','APP_PACKETS','DNS_QUERY_T','Type']
dataset = pd.read_csv("dataset.csv", header=0, names=names)


no_input = 160
seed = 2
X_train, X_test, y_train, y_test, X, y = pretreatment(dataset, no_input, seed)

evaluate_KNN(X_test, y_test, X_train, y_train)

# display_label(y)
# find_no_input_feature()
# find_no_nodes()

# vote(evaluate_random_forest(X_test, y_test, X_train, y_train), evaluate_neural(X_test, y_test, X_train, y_train, no_input, 50, 70, 50)[1], y_test)



