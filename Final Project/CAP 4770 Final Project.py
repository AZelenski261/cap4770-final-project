import pandas as pd
import numpy as np
import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import preprocessing

def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    ########################################## Load Data Sets ##########################################
    trainData = pd.read_csv('./titanic/train.csv',
        usecols=['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
        dtype={'PassengerId' : int, 'Survived' : int, 'Pclass' : int, 'Name' : str, 'Sex' : str, 'Age' : float, 'SibSp' : int, 'Parch' : int, 'Ticket' : str, 'Fare' : float, 'Cabin' : str, 'Embarked' : str}
    )
    testData = pd.read_csv('./titanic/test.csv',
        usecols=['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
        dtype={'PassengerId' : int, 'Pclass' : int, 'Name' : str, 'Sex' : str, 'Age' : float, 'SibSp' : int, 'Parch' : int, 'Ticket' : str, 'Fare' : float, 'Cabin' : str, 'Embarked' : str}
    )

    ############################################ Remove NaNs ###########################################
    #
    # NaNs in the Data
    # Test Dataset Size: 418, Train Dataset Size: 891
    # 'Age' - NaNs present in test (86) and train (177) data
    # 'Fare' - NaNs present in test (1) data
    # 'Cabin' - NaNs present in test (327) and train (687) data
    # 'Embarked' - NaNs present in train (2) data
    #
    # The NaNs in the trainData Age column are replaced by the median of the train Age
    trainDataAgeMedian = trainData['Age'].median(skipna=True)
    trainData['Age'] = trainData['Age'].fillna(trainDataAgeMedian).astype(int)

    # The NaNs in the testData Age column are replaced by the median of the test Age
    testDataAgeMedian = testData['Age'].median(skipna=True)
    testData['Age'] = testData['Age'].fillna(testDataAgeMedian).astype(int)

    # The NaNs in the testData Fare column are replaced by the median of the test Fare
    testDataFareMedian = testData['Fare'].median(skipna=True)
    testData['Fare'] = testData['Fare'].fillna(testDataFareMedian).astype(int)

    ####################### Create FamilySize and AgeClassInteraction attributes #######################
    trainData['AgeClassInteraction'] = trainData['Age'] * trainData['Pclass']
    testData['AgeClassInteraction'] = testData['Age'] * testData['Pclass']
    trainData['FamilySize'] = trainData['SibSp'] + trainData['Parch']
    testData['FamilySize'] = testData['SibSp'] + testData['Parch']

    ##################################### Determine Columns to Use #####################################
    # List of columns to use in creating the model and the data type
    #   'C' - Categorical
    #   'N' - Numeric
    #   'O' - Ordinal
    colList = [('Pclass', 'O'), ('Sex', 'C'), ('Age', 'N'), ('SibSp', 'N'), ('Parch', 'N'), ('Fare', 'N'), ('FamilySize', 'N'), ('AgeClassInteraction', 'N')]

    ######################################### Preprocess Data ##########################################
    trainDataArray = preprocessData(trainData, colList)
    testDataArray = preprocessData(testData, colList)

    ########################################## Create Model(s) #########################################
    classifiers = [
        ("Nearest Neighbors", KNeighborsClassifier()),
        ("SVM", SVC()),
        ("Gaussian Process", GaussianProcessClassifier()),
        ("Decision Tree", DecisionTreeClassifier()),
        ("Random Forest", RandomForestClassifier()),
        ("Neural Net", MLPClassifier()),
        ("AdaBoost", AdaBoostClassifier()),
        ("Naive Bayes", GaussianNB()),
        ("QDA", QuadraticDiscriminantAnalysis()),
    ]

    models = []
    for i in range(len(classifiers)):
        models.append((classifiers[i][0], classifiers[i][1].fit(trainDataArray, trainData['Survived'])))

    ########################################## Get Predictions #########################################
    summaryDF = testData[['PassengerId']].copy()
    resultDFs = [] # List of tuples of Model name [0] and resulting dataframe (columns PassengerId and Survived) [1]
    
    for model in models:
        resultDFs.append((model[0], getPredictions(model[1], testData['PassengerId'], testDataArray)))
        summaryDF[model[0] + ' Survived'] = resultDFs[-1][1]['Survived']
    
    print('\n\n')
    print(summaryDF.head(10))
    summaryDF.to_csv(path_or_buf='modelSummary ' + datetime.datetime.now().strftime("%Y-%m-%d at %H.%M.%S") + '.csv', header=True, index=False, mode='w')

    # Result DataFrame to .csv for Kaggle Submission
    #resultDFs['Insert Classifier Number'][1].to_csv(path_or_buf='kaggle_submission ' + datetime.datetime.now().strftime("%Y-%m-%d at %H.%M.%S") + '.csv', header=True, index=False, mode='w')



# For Numeric and Ordinal values, NOTHING is done,
#   For Categorical values, they are processed converted to Ordinal values
# Throws an error if NaNs are present in the column
def preprocessData(dataFrame: pd.DataFrame, columnsToProcess: list) -> np.array:
    tempData = []
    enc = preprocessing.OrdinalEncoder()

    for col in columnsToProcess:
        if (col[1] == 'C'):
            tempDataArray = enc.fit_transform(dataFrame[[col[0]]])
            for i in range(np.shape(tempDataArray)[1]):
                tempData.append(tempDataArray[:, i])
        else:
            tempData.append(dataFrame[col[0]].values)
    
    return np.stack(tempData, axis=1)



#Will ONLY work if the model has a predict function
def getPredictions(model, passengerIDsColumn: pd.Series, dataArrayToPredict: np.array) -> pd.DataFrame:
    results = []
    for i in range(len(dataArrayToPredict)):
        results.append(model.predict([dataArrayToPredict[i,:]])[0])
    return pd.concat([passengerIDsColumn, pd.DataFrame({'Survived' : results})], axis=1)

if __name__ == "__main__":
    main()