import PredictionModel as pm
import pandas as pd
from sklearn.model_selection import GridSearchCV

# To split the data set
from sklearn.model_selection import train_test_split

# To avoid warning from interfering the execution.
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    """Step 1: Visualize data"""
    data = pd.read_csv("Crop_recommendation.csv")
    data = data.dropna()  # drop None or NaN elements
    # print(data)

    """Step 2: Split the data set for Training and Testing"""
    # Separate features and target labels from raw data set.
    features = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    target = data['label']
    # Training set = 70% ; Testing set = 30%
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=0.20, train_size=0.80, random_state=2)

    """Step 3: Perform Model Fitting and Make Predictions"""
    models = pm.TrainingModels(Xtrain, Ytrain, Xtest)

    # lg_model, lg_pred = models.logisticRegressionModel()
    # gnb_model, gnb_pred = models.gaussianNaiveBayesModel()
    # knn_model, knn_pred = models.knearestNeighborModel()
    # sgd_model, sgd_pred = models.stochasticGradientDescentModel()

    """Step 4: Evaluate the Performance of each model"""
    performance = pm.PerformanceMeasures(Xtest, Ytest)
    #performance.plotConfusionMatrix(knn_model, 'KNN')
    #print('\n\nClassification Report - KNN')
    #performance.printClassificationReport(Y_pred=knn_pred)
    #performance.perf_measure(Ytest, knn_pred)

    """ Tuning the Models"""
    tuning = pm.TuningModels(Xtrain, Ytrain)
    tuning.tuneGNG()
    # tuning.tuneLogisticRegression()
    # tuning.tuneKnearestNeighbors()
    # tuning.tuneStochasticGradientDescent()

    # print("Logistic Regression")
    #lg_accuracy = performance.runAnalysis(lg_model, lg_pred)
    #
    # print("Gaussian NaiveBayes")
    # gnb_accuracy = performance.runAnalysis(gnb_model, gnb_pred)
    #
    # print("K nearest Neighbor")
    # knn_accuracy = performance.runAnalysis(knn_model, knn_pred)
    #
    # print("Stochastic Gradient Descent")
    #
    # sgd_accuracy = performance.runAnalysis(sgd_model, sgd_pred)
    #
    #
    #
    # """ Plot Accuracy Graph """
    # LGR = round(lg_accuracy, 2)
    # GNB = round(gnb_accuracy,2)
    # KNN = round(knn_accuracy, 2)
    # SGD = round(sgd_accuracy, 2)
    # SVM = 93.63 # Support Vector Machine
    # NN = 92.5 # Neural Net
    # DT = 98.86 # Decision Tree
    # RF = 98.86 # Random Forest
    # ADB = 99.31 # boosting on DT
    # GB = 98.18 # Gradient Boost - Decision Tree with Regression
    # XGB = ''
    #
    # import matplotlib.pyplot as plt
    # import matplotlib.axes as axes
    # import numpy as np
    #
    # # creating the dataset
    # labels = ['LGR', 'GNB', 'KNN', 'SGD', 'SVM','NN', 'DT', 'RF','ADB','GB']
    # y = [LGR, GNB, KNN, SGD, SVM, NN, DT, RF, ADB, GB]
    # xs = np.arange(len(labels))
    # fig = plt.figure(figsize=(10, 5))
    #
    # # creating the bar plot
    # plt.bar(xs, y, color='maroon',width=0.4)
    # plt.xticks(xs, labels=labels)
    # plt.xlabel("Models")
    # plt.ylabel("Accuracy")
    # plt.title("Performance of the Algorithms")
    # plt.show()

    
""" # PRINT ACCURACY

    print("\n\nAccuracy")
    print("Logistic Regression: ", lg_accuracy)
    print("Gaussian NaiveBayes: ", gnb_accuracy)
    print("K nearest Neighbor: ", knn_accuracy)
    print("Stochastic Gradient Descent: ", sgd_accuracy)
"""
