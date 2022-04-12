import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# importing machine learning models for prediction
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC, LinearSVC

def split_data(X, y):
    # Split the data into training, testing and validation sets (80%, 10%, 10%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=2)

    return X_train, X_test, X_val, y_train, y_test, y_val

def main():
    df = pd.read_csv('cleaned_data/cleaned_numeric_training_data.csv')
    print(df.head())

    labels = df['damage_grade']
    building_ids = df['building_id']
    training_data = df.drop(['building_id', 'damage_grade'], axis=1)

    X_train, X_test, X_val, y_train, y_test, y_val = split_data(training_data, labels)

    print('--------------Train Set--------------\n', X_train)
    print('--------------Test Set--------------\n', X_test)
    print('--------------Validation Set--------------\n', X_val)

    print('--------------Train Set Labels--------------\n', y_train)
    print('--------------Test Set Labels--------------\n', y_test)
    print('--------------Validation Set Labels--------------\n', y_val)

    # Run random forest
    print('Running random forest classifier')
    rf = RandomForestClassifier()
    print('Fitting random forest classifier')
    rf.fit(X_train, y_train)
    print('Making predictions')
    rf_pred = rf.predict(X_test)

    print('Creating confusion matrix for random forest classifier')
    rf_confusion_matrix = confusion_matrix(y_test, rf_pred)
    print(rf_confusion_matrix)
    print('Printing confusion matrix')
    ConfusionMatrixDisplay.from_predictions(y_test, rf_pred)
    plt.show()
    print('Calculating f1 score')
    print(f1_score(y_test, rf_pred, average='micro'))


if __name__ == '__main__':
    main()