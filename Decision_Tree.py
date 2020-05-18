import sys

import pandas as pd
import scipy
import numpy
import matplotlib
import pandas
import sklearn
import matplotlib.pyplot as plt
import random
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier






def decision_tree(X, Y, testSize, r):
    # Split data to training and testing 70:30
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testSize, random_state=r)
    # create the classifier
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # finally calculate Accuracy
    Accuracy = sklearn.metrics.accuracy_score(y_test, y_pred) * 100  # multiply * 100 for percentage
    print("and random split :", r, " Accuracy : ", Accuracy, "%")
    x = clf.tree_
    print("number of nodes in the tree: ", x.node_count)
    print("___________________________________________________________")
    return  Accuracy, x.node_count








def main():
    names = ["p", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14",
             "s15", "s16"]
    dataset = pandas.read_csv("house-votes-84.data.txt", names=names)
    # Separate between features_Data  and Target_Data
    X = dataset.drop("p", axis="columns")
    Y = dataset["p"]
    # deal with "?"
    for i in range(1, 17):
        if len(X[X.get("s" + str(i)) == "y"]) > len(X[X.get("s" + str(i)) == "n"]):
            X["s" + str(i)] = X["s" + str(i)].replace(to_replace="?", value="y")
        else:
            X["s" + str(i)] = X["s" + str(i)].replace(to_replace="?", value="n")

    # print(X)
    # Encoding dataset labels into numbers
    label_encoder = preprocessing.LabelEncoder()
    for i in range(1, 17):
        X["s" + str(i)] = label_encoder.fit_transform(X["s" + str(i)])

    # print(X["s1"].unique())
    # print(X)
    counter = 0
    n=0
    testSize = .3
    acc = [0,0,0,0,0]
    mean_acc=[0,0,0,0,0]
    treeSize = [0,0,0,0,0]
    mean_treeSize=[0,0,0,0,0]
    while counter < 5:
        print("Experiment", counter,":")
        while n<5:
            r = random.randrange(1, 435, 1)
            print( "(  trainingSize :", 1-testSize, ")\n")
            acc[n],treeSize[n]= decision_tree(X, Y, testSize, r)
            n += 1
        mean_acc[counter]=sum(acc)/len(acc)
        mean_treeSize[counter] =sum(treeSize) / len(treeSize)
        print("Max Accuracy is", max(acc), "\tmin accuracy is ",min(acc), "\tMean accuracy is",mean_acc[counter] , "\n")
        print("Max treeSize is", max(treeSize), "\tmin treeSize is ", min(acc), "\tMean treeSize is",mean_treeSize[counter], "\n")
        print("----------------------------------------------------------------------------------------------")
        counter += 1
        testSize += .1
        n=0
    #df1 = pd.DataFrame({'accuracy':mean_acc}, index=[0,1,2,3,4])
    plt.plot([70,60,50,40,30], mean_acc, color='red')
    plt.show()
    plt.plot([70, 60, 50, 40, 30], mean_treeSize, color='red')
    plt.show()
    # df1.plot()
    # plt.show()
    # df2 = pd.DataFrame({'numOFtreeSize': mean_treeSize}, index=[0, 1, 2, 3, 4])
    # df2.plot()






if __name__ == "__main__": main()



