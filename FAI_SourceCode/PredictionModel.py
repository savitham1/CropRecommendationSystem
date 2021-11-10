""" Import necessary libraries """
import matplotlib.pyplot as plt
import time
import joblib
import numpy as np
import os

plt.rc("font", size=14)

# Statistical data visualization library
import seaborn as sns

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# To avoid warning from interfering the execution.
import warnings

warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

class TrainingModels:
    def __init__(self, Xtrain, Ytrain, Xtest):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest = Xtest

    def logisticRegressionModel(self):

        # Model 1 = LogisticRegresssion
        # solver = lbfgs ; 95.227 = saga
        # solver = sag with penalty l2 better -> 95.4545
        # solver = newton-cg with l2 is the best with 97.5
        # solver = newton-cg ; penalty = l2 ; class-weight = 'balanced' gives 97.72%
        # For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss

        start = time.time()
        logRegModel = LogisticRegression(penalty='l2', solver='newton-cg', class_weight='balanced')
        logRegModel = logRegModel.fit(self.Xtrain, self.Ytrain)
        end = time.time()
        joblib.dump(logRegModel, "logRegJob.joblib")
        path = '/Users/savithamunirajaiah/Documents/All_Course_Database/NEU/Spring2021_Algo_FAI/FAI_CS5100/FinalProject/logRegJob.joblib'
        print(f"LR size: {np.round(os.path.getsize(path) / 1024 / 1024, 2)} MB")
        print("Time for DT ", end - start)
        # See test data to perform prediction
        logReg_Pred = logRegModel.predict(self.Xtest)
        return logRegModel, logReg_Pred

    def gaussianNaiveBayesModel(self):
        from sklearn.naive_bayes import GaussianNB
        # Model 2 = Gaussian Naive Bayes
        start = time.time()
        GaussNBayes = GaussianNB()
        GaussNBayes = GaussNBayes.fit(self.Xtrain, self.Ytrain)
        end = time.time()
        joblib.dump(GaussNBayes, "GNBjob.joblib")
        path = '/Users/savithamunirajaiah/Documents/All_Course_Database/NEU/Spring2021_Algo_FAI/FAI_CS5100/FinalProject/GNBjob.joblib'
        print(f"LR size: {np.round(os.path.getsize(path) / 1024 / 1024, 2)} MB")
        print("Time for GNB ", end - start)
        GNBayes_Pred = GaussNBayes.predict(self.Xtest)

        return GaussNBayes, GNBayes_Pred

    def knearestNeighborModel(self):
        from sklearn.neighbors import KNeighborsClassifier
        # Model 3 = K-nearest Neighbor
        # neighbor = 25,24 - 97.5 % 23 = 97.27%
        # Manhattan dist or minkowski with p = 1 is give 97.727%
        start = time.time()
        knn = KNeighborsClassifier(n_neighbors= 10, algorithm='auto', weights='uniform', metric='manhattan')
        knn = knn.fit(self.Xtrain, self.Ytrain)
        end = time.time()
        joblib.dump(knn, "knnjob.joblib")
        path = '/Users/savithamunirajaiah/Documents/All_Course_Database/NEU/Spring2021_Algo_FAI/FAI_CS5100/FinalProject/knnjob.joblib'
        print(f"LR size: {np.round(os.path.getsize(path) / 1024 / 1024, 2)} MB")
        print("Time for knn ", end - start)
        knn_Pred = knn.predict(self.Xtest)
        return knn, knn_Pred

    def stochasticGradientDescentModel(self):
        from sklearn.linear_model import SGDClassifier
        # Model 4 = Stochastic Gradient Descent Classifier
        # loss = modified_huber ; 75.90
        # penalty l1/elasticnet works well for sparse graph
        start = time.time()
        sgd = SGDClassifier(loss='modified_huber', penalty='l1', class_weight='balanced')
        sgd = sgd.fit(self.Xtrain, self.Ytrain)
        end = time.time()
        joblib.dump(sgd, "sgdjob.joblib")
        path = '/Users/savithamunirajaiah/Documents/All_Course_Database/NEU/Spring2021_Algo_FAI/FAI_CS5100/FinalProject/sgdjob.joblib'
        print(f"sgd size: {np.round(os.path.getsize(path) / 1024 / 1024, 2)} MB")
        print("Time for sgd ", end - start)
        sgd_Pred = sgd.predict(self.Xtest)
        return sgd, sgd_Pred

class TuningModels:
    def __init__(self, Xtrain, Ytrain):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain

    def tuneLogisticRegression(self):
        model = LogisticRegression()
        parameters1 = {'penalty': ['l1', 'l2'],
                      'solver': ['newton-cg'],
                      'class_weight': ['balanced', 'dict', 'None']
                     }
        parameters2 = {'penalty': ['l1', 'l2'],
                      'solver': [ 'sag'],
                      'class_weight': ['balanced', 'dict', 'None']
                      }
        parameters3 = {'penalty': ['l1', 'l2'],
                      'solver': ['saga'],
                      'class_weight': ['balanced', 'dict', 'None']
                      }
        parameters4 = {'penalty': ['l1', 'l2'],
                      'solver': ['lbfgs'],
                      'class_weight': ['balanced', 'dict', 'None']
                      }
        searcher = GridSearchCV(model, parameters1)
        searcher.fit(self.Xtrain, self.Ytrain.values.ravel())
        p1 = searcher.best_score_

        searcher = GridSearchCV(model, parameters2)
        searcher.fit(self.Xtrain, self.Ytrain.values.ravel())
        p2 = searcher.best_score_

        searcher = GridSearchCV(model, parameters3)
        searcher.fit(self.Xtrain, self.Ytrain.values.ravel())
        p3 = searcher.best_score_

        searcher = GridSearchCV(model, parameters4)
        searcher.fit(self.Xtrain, self.Ytrain.values.ravel())
        p4 = searcher.best_score_
        # print("Best param:", searcher.best_params_)
        # print("Best accuracy", searcher.best_score_)

        """ Find the best"""
        parameters = {'penalty': ['l1', 'l2'],
                      'solver': ['newton-cg','sag', 'saga','lbfgs'],
                      'class_weight': ['balanced', 'dict', 'None']
                      }
        searcher = GridSearchCV(model, parameters)
        searcher.fit(self.Xtrain, self.Ytrain.values.ravel())
        p = searcher.best_score_
        print("Best param:", searcher.best_params_)
        print("Best accuracy", searcher.best_score_)

        import matplotlib.pyplot as plt
        import matplotlib.axes as axes
        import numpy as np

        # creating the dataset
        labels = ['newton-cg', 'sag', 'saga', 'lbfgs']
        y = [p1, p2, p3, p4]
        xs = np.arange(len(labels))
        fig = plt.figure(figsize=(10, 5))

        # creating the bar plot
        plt.bar(xs, y, color='maroon',width=0.3)
        plt.xticks(xs, labels=labels)
        plt.xlabel("Solvers")
        plt.ylabel("Accuracy")
        plt.title("Performance under different solvers")
        plt.show()

    def tuneGNG(self):
        from sklearn.naive_bayes import GaussianNB
        from sklearn.preprocessing import PowerTransformer
        import numpy as np
        model = GaussianNB()
        parameters1 = {'var_smoothing': np.logspace(0,-9, num=1)
                       }
        parameters2 = {'var_smoothing': np.logspace(0,-9, num=200)
                       }
        parameters3 = {'var_smoothing': np.logspace(0,-9, num=11)
                       }
        parameters4 = {'var_smoothing': np.logspace(0,-9, num=100)
                       }
        searcher = GridSearchCV(model, parameters1)
        searcher.fit(self.Xtrain, self.Ytrain.values.ravel())
        p1 = searcher.best_score_

        searcher = GridSearchCV(model, parameters2)
        searcher.fit(self.Xtrain, self.Ytrain.values.ravel())
        p2 = searcher.best_score_

        searcher = GridSearchCV(model, parameters3)
        searcher.fit(self.Xtrain, self.Ytrain.values.ravel())
        p3 = searcher.best_score_

        searcher = GridSearchCV(model, parameters4)
        searcher.fit(self.Xtrain, self.Ytrain.values.ravel())
        p4 = searcher.best_score_

        import matplotlib.pyplot as plt
        import matplotlib.axes as axes
        import numpy as np

        # creating the dataset
        labels = ['s1', 's2', 's3', 's4']
        y = [p1, p2, p3, p4]
        xs = np.arange(len(labels))
        fig = plt.figure(figsize=(10, 5))

        # creating the bar plot
        plt.bar(xs, y, color='maroon', width=0.3)
        plt.xticks(xs, labels=labels)
        plt.xlabel("var_smoothing")
        plt.ylabel("Accuracy")
        plt.title("Performance under different var_smoothing")
        plt.show()


        parameters = {'var_smoothing': np.logspace(0,-9, num=100)}
        searcher = GridSearchCV(model, parameters)
        searcher.fit(self.Xtrain, self.Ytrain.values.ravel())
        p = searcher.best_score_
        print("Best param:", searcher.best_params_)
        print("Best accuracy", searcher.best_score_)

    def tuneKnearestNeighbors(self):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()
        parameters1 = {'n_neighbors' :[100],
                        'metric': ['minkowski', 'manhattan']
                     }

        parameters2 = {'n_neighbors': [10],
                        'metric':['minkowski', 'manhattan']
                     }

        parameters3 = {'n_neighbors':[25],
                        'metric':['minkowski', 'manhattan']
                     }

        parameters4 = {'n_neighbors' :[50],
                        'metric':['minkowski', 'manhattan']
                     }

        searcher = GridSearchCV(model, parameters1)
        searcher.fit(self.Xtrain, self.Ytrain.values.ravel())
        p1 = searcher.best_score_

        searcher = GridSearchCV(model, parameters2)
        searcher.fit(self.Xtrain, self.Ytrain.values.ravel())
        p2 = searcher.best_score_

        searcher = GridSearchCV(model, parameters3)
        searcher.fit(self.Xtrain, self.Ytrain.values.ravel())
        p3 = searcher.best_score_

        searcher = GridSearchCV(model, parameters4)
        searcher.fit(self.Xtrain, self.Ytrain.values.ravel())
        p4 = searcher.best_score_

        """ Find the best"""
        parameters = {'n_neighbors': [50, 100, 10, 25, 24],
                       'metric': ['minkowski', 'manhattan']
                       }
        searcher = GridSearchCV(model, parameters)
        searcher.fit(self.Xtrain, self.Ytrain.values.ravel())
        p = searcher.best_score_
        print("Best param:", searcher.best_params_)
        print("Best accuracy", searcher.best_score_)

        """ Plot the graph """
        import matplotlib.pyplot as plt
        import matplotlib.axes as axes
        import numpy as np


        # creating the dataset
        labels = ['100', '10', '25', '50']
        y = [p1, p2, p3, p4]
        xs = np.arange(len(labels))
        fig = plt.figure(figsize=(10, 5))

        # creating the bar plot
        plt.bar(xs, y, color='maroon',width=0.3)
        plt.xticks(xs, labels=labels)
        plt.xlabel("Number of Neighbors")
        plt.ylabel("Accuracy")
        plt.title("Performance under different k - neighbors")
        plt.show()


    def tuneStochasticGradientDescent(self):
        from sklearn.linear_model import SGDClassifier

        model = SGDClassifier()
        parameters1 = {'loss':['modified_huber'],
                       'penalty':['l1', 'l2'],
                       'class_weight':['balanced', 'dict', 'None']
                       }

        parameters2 = {'loss':['log'],
                       'penalty':['l1', 'l2'],
                       'class_weight':['balanced', 'dict', 'None']
                       }

        parameters3 = {'loss':['hinge'],
                       'penalty':['l1', 'l2'],
                       'class_weight':['balanced', 'dict', 'None']
                       }

        parameters4 = {'loss':['perceptron'],
                       'penalty':['l1', 'l2'],
                       'class_weight':['balanced', 'dict', 'None']
                       }

        searcher = GridSearchCV(model, parameters1)
        searcher.fit(self.Xtrain, self.Ytrain.values.ravel())
        p1 = searcher.best_score_
        print(p1)
        searcher = GridSearchCV(model, parameters2)
        searcher.fit(self.Xtrain, self.Ytrain.values.ravel())
        p2 = searcher.best_score_
        print(p2)
        searcher = GridSearchCV(model, parameters3)
        searcher.fit(self.Xtrain, self.Ytrain.values.ravel())
        p3 = searcher.best_score_
        print(p3)
        searcher = GridSearchCV(model, parameters4)
        searcher.fit(self.Xtrain, self.Ytrain.values.ravel())
        p4 = searcher.best_score_
        print(p4)

        """ Find the best"""
        parameters = {'loss': ['modified_huber','log', 'hinge', 'perceptron'],
                       'penalty': ['l1', 'l2'],
                       'class_weight': ['balanced', 'dict', 'None']
                       }
        searcher = GridSearchCV(model, parameters)
        searcher.fit(self.Xtrain, self.Ytrain.values.ravel())
        p = searcher.best_score_
        print("Best param:", searcher.best_params_)
        print("Best accuracy", searcher.best_score_)

        """ Plot the graph """
        import matplotlib.pyplot as plt
        import matplotlib.axes as axes
        import numpy as np

        # creating the dataset
        labels = ['modified_huber','log', 'hinge', 'perceptron']
        y = [p1, p2, p3, p4]
        xs = np.arange(len(labels))
        fig = plt.figure(figsize=(10, 5))

        # creating the bar plot
        plt.bar(xs, y, color='maroon', width=0.3)
        plt.xticks(xs, labels=labels)
        plt.xlabel("Different loss functions")
        plt.ylabel("Accuracy")
        plt.title("Performance under different loss function")
        plt.show()


class PerformanceMeasures:
    def __init__(self, Xtest, Ytest):
        # Count the hits and misses of the model
        from sklearn.metrics import confusion_matrix as cm
        from sklearn.metrics import plot_confusion_matrix as plotcm  # Display the matrix
        # Measure the performance
        from sklearn.metrics import classification_report as CLreport

        self.Xtest = Xtest
        self.Ytest = Ytest
        self.confusionMatrix = cm
        self.plot_confusionMatrix = plotcm
        self.classificationReport = CLreport

    def plotConfusionMatrix(self, model, modelnamefordisplay):
        # Plot confusion matrix - Display
        #print(modelnamefordisplay)
        self.plot_confusionMatrix(model, self.Xtest, self.Ytest, cmap=plt.cm.Blues)
        plt.xticks(rotation=60)
        plt.title(modelnamefordisplay)
        plt.show()

    # Confusion matrix is a performance measurement for ML classification.
    # [0,0] TP = True Positive;  [0,1] FP = False Positive;
    # [1,0] FN = False Negative; [1,1] TN = True Negative;
    def printConfusionMatrix(self, Y_pred):
        # Print the confusion matrix
        print(self.confusionMatrix(y_pred=Y_pred, y_true=self.Ytest))

    def Accuracy(self, model):
        accuracy = model.score(self.Xtest, self.Ytest)
        print(accuracy)
        return accuracy

    def printClassificationReport(self, Y_pred):
        print(self.classificationReport(y_true=self.Ytest, y_pred=Y_pred))

    def runAnalysis(self, model, Y_pred):
        print(self.classificationReport(self.Ytest, Y_pred))
        print(self.confusionMatrix(Y_pred, self.Ytest))
        score = self.Accuracy(model) * 100  # converting into percentage
        print("Accuracy: ", score)
        return score

    def plotAccuracy(self, x=[], y=[]):
        plt.plot(x,y)
        plt.show()


"""
    def perf_measure(self, y_actual, y_pred):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_pred)):
            if y_actual[i] == y_pred[i] == 1:
                TP += 1
            if y_pred[i] == 1 and y_actual[i] != y_pred[i]:
                FP += 1
            if y_actual[i] == y_pred[i] == 0:
                TN += 1
            if y_pred[i] == 0 and y_actual[i] != y_pred[i]:
                FN += 1

        return (TP, FP, TN, FN)
"""
