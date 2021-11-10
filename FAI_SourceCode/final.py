# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os

import np as np
import time
import joblib
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier # Import Random Forest Classifier
from sklearn.ensemble import AdaBoostClassifier # Import adaboost Classifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV  # Import train_test_split function

from sklearn import metrics as met
from sklearn import tree
import matplotlib.pyplot as matplot



if __name__ == '__main__':
    modelWithAccuray = {}

    columnNames = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
    featureCol = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    targetCol = ['label']

    trainingData = pd.read_csv('C:/NEU/FAI/Project/train_dataset.csv', names=columnNames, skiprows=1)
    trainFeatures = trainingData[featureCol]
    trainTarget = trainingData[targetCol]

    testData = pd.read_csv('C:/NEU/FAI/Project/test_dataset.csv', names=columnNames, skiprows=1)
    testFeatures = testData[featureCol]
    testTarget = testData[targetCol]

    validatedata = pd.read_csv('C:/NEU/FAI/Project/validation_dataset.csv', names=columnNames, skiprows=1)
    validateFeatures = validatedata[featureCol]
    validateTarget = validatedata[targetCol]



    ## Decision tree

    model = DecisionTreeClassifier()

    parameters = {'criterion': ['gini', 'entropy'],
                  'max_depth': [10, 50, 100, 200],
                  'min_samples_split': [1, 5, 10],
                  'max_features': [3, 5, 7, 100],
                  'min_samples_leaf': [0.5, 1, 5]}

    searcher = GridSearchCV(model, parameters)

    searcher.fit(validateFeatures, validateTarget.values.ravel())
    print("Best param:", searcher.best_params_)
    print("Best accuracy",searcher.best_score_)

    clf3 = DecisionTreeClassifier(criterion='gini', max_depth=100, max_features=5, min_samples_leaf=1,
                                  min_samples_split=5)
    start = time.time()
    clf3.fit(trainFeatures, trainTarget.values.ravel())
    end = time.time()
    joblib.dump(clf3, "DT_100_trees.joblib")
    print(f"DT size: {np.round(os.path.getsize('RandomForest_100_trees.joblib') / 1024 / 1024, 2)} MB")
    print("Time for DT ", end - start)
    pred = clf3.predict(testFeatures)
    accuracy = accuracy_score(testTarget, pred)
    print(accuracy)

    listOfacc = []
    for i in range(1, 8, 1):
        clf = DecisionTreeClassifier(max_features=i)
        clf.fit(trainFeatures, trainTarget.values.ravel())
        pred = clf.predict(testFeatures)
        accuracy = accuracy_score(testTarget, pred)
        listOfacc.append(accuracy)

    vals = list(range(1, 8, 1))

    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(vals, listOfacc, color='orange', width=0.2)

    plt.xlabel("Number of features")
    plt.ylabel("Accuracy")
    plt.title("Variation of accuracy with features")
    plt.show()

    ##RANDOM FOREST
    model = RandomForestClassifier()

    parameters = {'criterion':['gini','entropy'],
                  'max_depth': [10, 50, 100, 200],
                  'min_samples_split':[1, 5, 10],
                  'max_features': [3, 5, 7, 100],
                  'min_samples_leaf':[0.5, 1,  5],
                  'n_estimators': [5, 10, 15, 20, 25]}

    searcher = GridSearchCV(model, parameters)

    searcher.fit(validateFeatures, validateTarget.values.ravel())
    print("Best param:", searcher.best_params_)
    print("Best accuracy",searcher.best_score_)

    clf = RandomForestClassifier(criterion= 'entropy',
                                 max_depth= 50,
                                 max_features=3,
                                 min_samples_leaf= 1,
                                 min_samples_split= 5,
                                 n_estimators= 15)

    start = time.time()
    clf.fit(trainFeatures, trainTarget.values.ravel())
    end = time.time()
    joblib.dump(clf, "DT_100_trees.joblib")
    print(f"RF size: {np.round(os.path.getsize('RandomForest_100_trees.joblib') / 1024 / 1024, 2)} MB")
    print("Time for RF ", end - start)
    pred = clf.predict(testFeatures)
    accuracy = accuracy_score(testTarget, pred)
    print(accuracy)

    vals = [10, 50, 100, 200, 1000]
    listOfacc = []
    for i in range(1,24,3):
        clf = RandomForestClassifier(n_estimators=i)
        clf.fit(trainFeatures, trainTarget.values.ravel())
        pred = clf.predict(testFeatures)
        accuracy = accuracy_score(testTarget, pred)
        listOfacc.append(accuracy)

    vals = list(range(1, 24, 3))

    fig = plt.figure(figsize=(10, 5))
    plt.ylim(0.8, 1)

    # creating the bar plot
    plt.bar(vals, listOfacc, color='orange', width=0.2)

    plt.xlabel("Number of Decision Trees")
    plt.ylabel("Accuracy")
    plt.title("Variation of accuracy with number of trees")
    plt.show()



    ##ADABOOST

    clf = RandomForestClassifier(criterion= 'entropy',
                                 max_depth= 50,
                                 max_features=3,
                                 min_samples_leaf= 1,
                                 min_samples_split= 5,
                                 n_estimators= 15)
    clf.fit(trainFeatures, trainTarget.values.ravel())
    pred = clf.predict(testFeatures)
    accuracy = accuracy_score(testTarget, pred)
    # print(accuracy)

    clf2 =  svm.SVC(gamma= 0.1, C=1)
    clf2.fit(trainFeatures, trainTarget.values.ravel())
    clf2.predict(testFeatures)

    clf3 = DecisionTreeClassifier(criterion='gini', max_depth=100, max_features=5, min_samples_leaf=1, min_samples_split=5)
    clf3.fit(trainFeatures, trainTarget.values.ravel())
    clf3.predict(testFeatures)


    parameters = {'base_estimator':[clf,clf2, clf3],
                  'learning_rate':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
    model = AdaBoostClassifier()

    searcher = GridSearchCV(model, parameters)

    searcher.fit(validateFeatures, validateTarget.values.ravel())
    print("Best param:", searcher.best_params_)
    print("Best accuracy",searcher.best_score_)


    adaClf = AdaBoostClassifier(learning_rate=0.6, base_estimator=clf)
    start = time.time()
    adaClf.fit(trainFeatures, trainTarget.values.ravel())
    end = time.time()
    joblib.dump(adaClf, "DT_100_trees.joblib")
    print(f"AB size: {np.round(os.path.getsize('RandomForest_100_trees.joblib') / 1024 / 1024, 2)} MB")
    print("Time for AB ", end - start)
    pred = adaClf.predict(testFeatures)
    accuracy = accuracy_score(testTarget, pred)
    print(accuracy)


    listOfacc = []

    learningRate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for i in learningRate:
        clf = AdaBoostClassifier(learning_rate=i)
        clf.fit(trainFeatures, trainTarget.values.ravel())
        pred = clf.predict(testFeatures)
        accuracy = accuracy_score(testTarget, pred)
        listOfacc.append(accuracy)

    vals = learningRate

    fig = plt.figure(figsize=(10, 5))
    # plt.ylim(0.8,1)

    # creating the bar plot
    plt.bar(vals, listOfacc, color='orange', width = 0.01)

    plt.xlabel("Learning rate")
    plt.ylabel("Accuracy")
    plt.title("Variation of accuracy with learning rate")
    plt.show()




    #Gradient Boost

    parameters = {
                  'learning_rate':[0.1,0.5, 1],
                    'n_estimators' : [10,50,100],
                    'max_depth': [3, 5],
                    'criterion' : ["friedman_mse", "mse", "mae"]}
    model = GradientBoostingClassifier()

    searcher = GridSearchCV(model, parameters)

    searcher.fit(trainFeatures, trainTarget.values.ravel())
    print("Best param:", searcher.best_params_)
    print("Best accuracy",searcher.best_score_)


    clf3 = GradientBoostingClassifier(max_depth=8, learning_rate=1, n_estimators=100, criterion='friedman_mse')
    start = time.time()
    clf3.fit(trainFeatures, trainTarget.values.ravel())


    end = time.time()
    joblib.dump(clf3, "DT_100_trees.joblib")
    print(f"GB size: {np.round(os.path.getsize('DT_100_trees.joblib') / 1024 / 1024, 2)} MB")
    print("Time for GB ", end-start)
    clf3.predict(testFeatures)

    listOfacc = []

    learningRate = [2, 3, 4, 5]
    for i in learningRate:
        clf = GradientBoostingClassifier(learning_rate=1, max_depth=i, n_estimators=100, criterion='friedman_mse')
        clf.fit(trainFeatures, trainTarget.values.ravel())
        pred = clf.predict(testFeatures)
        accuracy = accuracy_score(testTarget, pred)
        listOfacc.append(accuracy)

    vals = learningRate

    fig = plt.figure(figsize=(10, 5))
    # plt.ylim(0.8,1)

    # creating the bar plot
    plt.bar(vals, listOfacc, color='orange', width=0.01)

    plt.xlabel("Depth of tree")
    plt.ylabel("Accuracy")
    plt.title("Variation of accuracy with depth of tree")
    plt.show()


    ##XGBoost
    parameters = {'booster': ['gbtree', 'gblinear', 'dart'],
                  'learning_rate': [0.2, 1],
                  'gamma': [0, 1, 5],
                  'max_depth': [6, 8, 50, 100],
                  'reg_lambda': [0, 1, 3]}


    y_label_train = list(np.array(trainTarget).ravel())
    y_label_validate = list(np.array(validateTarget).ravel())
    y_label_test = list(np.array(testTarget).ravel())

    label_encoder = LabelEncoder()
    y_label_train = label_encoder.fit_transform(y_label_train)
    y_label_validate = label_encoder.fit_transform(y_label_validate)
    y_label_test = label_encoder.fit_transform(y_label_test)

    model = xgb.XGBRegressor()

    searcher = GridSearchCV(model, parameters)
    searcher.fit(trainFeatures, y_label_train)

    print("Best param:", searcher.best_params_)
    print("Best accuracy", searcher.best_score_)

    start = time.time()
    model = xgb.XGBRegressor(max_depth=8, booster='dart', gamma=5, learning_rate=0.2, reg_lambda=1)
    model.fit(trainFeatures, y_label_train)
    end = time.time()
    joblib.dump(model, "DT_100_trees.joblib")
    print(f"XGB size: {np.round(os.path.getsize('DT_100_trees.joblib') / 1024 / 1024, 2)} MB")
    print("Time for XGB ", end - start)

    listOfacc = []

    learningRate = [0,1,2,3,4,5]
    for i in learningRate:
        clf = xgb.XGBRegressor(gamma=5, reg_lambda=i)
        clf.fit(trainFeatures, y_label_train)
        preds = clf.predict(testFeatures)
        predictions = [round(value) for value in preds]

        accuracy = accuracy_score(y_label_test, predictions)
        listOfacc.append(accuracy)

    vals = learningRate

    fig = plt.figure(figsize=(10, 5))
    plt.ylim(0.6,0.8)

    # creating the bar plot
    plt.bar(vals, listOfacc, color='orange', width = 0.01)

    plt.xlabel("Lambda")
    plt.ylabel("Accuracy")
    plt.title("Variation of accuracy with regularization parameter")
    plt.show()

    ## final time plot

    time = {}

    time['KNN'] = 0.0056
    time['GNB'] = 0.0068
    time['LR'] = 3.51
    time['NN'] = 116.01
    time['SVM'] = 0.18
    time['DT'] = 0.0059
    time['RF'] = 0.0538
    time['AB'] = 0.162
    time['GB'] = 12.74
    time['XGB'] = 1.63
    time['SGD'] = 0.1122

    keys = time.keys()
    values = time.values()
    plt.bar(keys, values, color='orange')
    plt.ylim(0,13)
    plt.xlabel("Algorithm")
    plt.ylabel("Time taken")
    plt.title("Algorithm v/s time taken")
    plt.xticks(rotation=45)
    plt.show()



    ## Memory

    memory = {}

    memory['KNN'] = 230
    memory['GNB'] = 4
    memory['LR'] = 3
    memory['SVM'] = 380
    memory['DT'] = 20
    memory['RF'] = 330
    memory['AB'] = 80
    memory['GB'] = 42670
    memory['XGB'] = 10800
    memory['SGD'] = 4

    keys = memory.keys()
    values = memory.values()
    plt.bar(keys, values, color='orange')
    plt.ylim(0, 400)
    plt.xlabel("Algorithm")
    plt.ylabel("Memory usage")
    plt.title("Algorithm v/s memory usage")
    plt.xticks(rotation=45)
    plt.show()


## Memory

    accuracy = {}

    accuracy['KNN'] = 97.95
    accuracy['GNB'] = 99.09
    accuracy['LR'] = 98.69
    accuracy['NN'] = 97.5
    accuracy['SVM'] = 96.5
    accuracy['DT'] = 98.86
    accuracy['RF'] = 99.09
    accuracy['AB'] = 99.24
    accuracy['GB'] = 98.78
    accuracy['XGB'] = 96.5
    accuracy['SGD'] = 80.05

    keys = accuracy.keys()
    values = accuracy.values()
    plt.bar(keys, values, color='orange')
    plt.ylim(75,100)
    plt.xlabel("Algorithm")
    plt.ylabel("Accuracy")
    plt.title("Algorithm v/s accuracy")
    plt.xticks(rotation=45)
    plt.show()

