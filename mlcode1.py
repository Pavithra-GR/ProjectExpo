import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib
import sys
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def mypredict(s,t,sp,c,ti):
    data = pd.read_csv(r'C:\Users\ELCOT\Documents\project expo\traffic_flow_data.csv')
    # Define feature columns and target column
    X = data[['source','target', 'speed','congestion_factor']]
    y = data['vehicles']
    f=['node','target', 'speed','congestion_factor']

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    # Create decision tree classifier
    clf = DecisionTreeClassifier(random_state=100)

    # Train the classifier
    clf.fit(X_train.values, y_train.values)

    # Predicting the response for test dataset
    y_pred = clf.predict(X_test.values)

    # Printing the accuracy of the decision tree classifier
    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    #tree.plot_tree(clf, feature_names=f)
    #print("Confusion Matrix: \n",confusion_matrix(y_test, y_pred))

    veh=clf.predict([[s,t,sp,c]])
    veh=int(veh[0])
    #print(veh)

    # Load dataset
    data = pd.read_csv(r'C:\Users\ELCOT\Documents\project expo\newrecord.csv')


    # Define feature columns and target column
    X = data[['duration','vehicles']]
    y = data['mode_of_transport']
    f=['duration','vehicles']

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    # Create decision tree classifier
    clf = DecisionTreeClassifier(random_state=100)

    # Train the classifier
    clf.fit(X_train.values, y_train.values)

    # Predicting the response for test dataset
    y_pred = clf.predict(X_test.values)

    # Printing the accuracy of the decision tree classifier
    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    #tree.plot_tree(clf, feature_names=f)
    #print("Confusion Matrix: \n",confusion_matrix(y_test, y_pred))
    nv=clf.predict([[veh,ti]])
    #print(nv[0])
    return nv[0]
d=mypredict(20,30,6.90,2.4567,45)
print(d)