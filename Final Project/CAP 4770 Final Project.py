import warnings
warnings.filterwarnings('ignore')
    # Supresses Neural Net Warning "ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet."
    # Supresses QDA Warning "UserWarning: Variables are collinear"

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
from sklearn import model_selection

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

    # copy of datasets, originals may be needed for reference
    trainCopy = trainData.copy(deep = True)
    testCopy = testData.copy(deep = True)
    datasets = [trainCopy, testCopy]

    ######################################### Engineer "Title" #########################################
    # since Name is a complete feature, we will use it to engineer the "Title" feature
    for dataset in datasets:
        dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    
    print("Titles in TRAIN:")
    print(datasets[0]['Title'].value_counts())
    print("-------------------")
    print("Titles in TEST:")
    print(datasets[1]['Title'].value_counts())

    # replace "rare" (fewer than 10 instances) and foreign titles with english equivalent
    for dataset in datasets:
        dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
        dataset['Title'] = dataset['Title'].replace(['Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 
                                                    'Sir', 'Jonkheer', 'Dona'], 
                                                    'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
    print("\n\nTitles in TRAIN:")
    print(datasets[0]['Title'].value_counts())
    print("-------------------")
    print("Titles in TEST:")
    print(datasets[1]['Title'].value_counts())

    ############################################ Remove NaNs ###########################################
    #
    # NaNs in the Data
    # Test Dataset Size: 418, Train Dataset Size: 891
    # 'Age' - NaNs present in test (86) and train (177) data
    # 'Fare' - NaNs present in test (1) data
    # 'Cabin' - NaNs present in test (327) and train (687) data
    # 'Embarked' - NaNs present in train (2) data
    
    # find incomplete columns
    print("\n\nSums of incomplete TRAINING values:")
    print(datasets[0].isnull().sum())
    print("------------------------------")
    print("Sums of incomplete TEST values:")
    print(datasets[1].isnull().sum())

    for dataset in datasets:
        # because a number decks housed first-, second-, and third-class passengers, 
        # we doubt the significance of the Cabin feature and will not attempt to complete it
        dataset.drop('Name', axis = 1, inplace = True)
        dataset.drop('Cabin', axis = 1, inplace = True)
        dataset.drop('Ticket', axis = 1, inplace = True)          # Ticket also appears to be insignificant
        dataset.drop('PassengerId', axis = 1, inplace = True)     # as does PassengerId

    # now that we have the Title attribute, we will complete the Age feature using the
    # median age associated with each title, and complete the rest of the features
    for dataset in datasets:
        dataset['Age'] = dataset.groupby('Title', as_index = True)['Age'].apply(lambda age: age.fillna(age.median()))
        dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
        dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    ####################################### Feature Engineering ########################################
    # FamilySize = siblings + spouse + parents + children
    # AgeClassInteraction = product of age and passenger class
    for dataset in datasets:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        dataset['AgeClassInteraction'] = dataset['Age'] * dataset['Pclass']

    ##################################### Determine Columns to Use #####################################
    # List of columns to use in creating the model and the data type
    #   'C' - Categorical
    #   'N' - Numeric
    #   'O' - Ordinal
    colList = [('Pclass', 'O'), ('Sex', 'C'), ('Age', 'N'), ('SibSp', 'N'), ('Parch', 'N'), ('Fare', 'N'), ('Embarked', 'C'), ('Title', 'C'), ('FamilySize', 'N'), ('AgeClassInteraction', 'N')]

    ######################################### Preprocess Data ##########################################
    preprocessedDatasets = [preprocessData(dataset, colList) for dataset in datasets]

    ########################################## Create Model(s) #########################################
    # Inspiration for Classifier Models was taken from the following page
    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    classifiers = [
        ("Default Nearest Neighbors", KNeighborsClassifier()),
        ("10 Nearest Neighbors (Brute)", KNeighborsClassifier(n_neighbors=10, algorithm='brute')),
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
        models.append((classifiers[i][0], classifiers[i][1].fit(preprocessedDatasets[0], datasets[0]['Survived'])))

    ########################################## Get Predictions #########################################
    summaryDF = testData[['PassengerId']].copy()
    resultDFs = [] # List of tuples of Model name [0] and resulting dataframe (columns PassengerId and Survived) [1]
    
    for model in models:
        resultDFs.append((model[0], getPredictions(model[1], testData['PassengerId'], preprocessedDatasets[1])))
        summaryDF[model[0]] = resultDFs[-1][1]['Survived']
    
    print('\n\n')
    print(summaryDF.head(10))
    summaryDF.to_csv(path_or_buf='modelSummary ' + datetime.datetime.now().strftime("%Y-%m-%d at %H.%M.%S") + '.csv', header=True, index=False, mode='w')

    print('\n\nScoring Models:')
    for model in models:
        cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)
        cv_results = model_selection.cross_validate(model[1], preprocessedDatasets[0], trainCopy['Survived'], cv  = cv_split)
        score = cv_results['test_score'].max()
        print('    ' + model[0] + ': ' + str(score))
    print()
    # Result DataFrame to .csv for Kaggle Submission
    #resultDFs['Insert Classifier Number'][1].to_csv(path_or_buf='kaggle_submission ' + datetime.datetime.now().strftime("%Y-%m-%d at %H.%M.%S") + '.csv', header=True, index=False, mode='w')



# For Numeric and Ordinal values, NOTHING is done, for Categorical values, they are converted to Ordinal values
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
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