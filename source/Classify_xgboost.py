import numpy as np
import pandas as pd
from sklearn import metrics
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def xgbpredictone():
    filepath = r'C:\Users\USER\Documents\PYTHON\MyDL\CNN\project_Seismic Signal Classification\data\example36.csv'
    modelpath = r'C:\Users\USER\Documents\PYTHON\MyDL\CNN\project_Seismic Signal Classification\data\xgbmodel_36.pkl'

    try:
        data = pd.read_csv(filepath, header=None, sep=',')
        data = data.dropna()
        
        x_test = data.iloc[:, 1:].values 
        y_test = data.iloc[:, 0]

        model = joblib.load(modelpath)
        y_test_pred = model.predict(x_test)
        
        test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
        print('--- XGBoost Model 36 Sonuçları ---')
        print('XGBoost test Accuracy: {:.6f}'.format(test_accuracy))
        print('Misclassified samples: {}'.format((y_test != y_test_pred).sum()))
    except Exception as e:
        print(f"Model 36 xətası: {e}")

def xgbpredicttwo():
    filepath = r'C:\Users\USER\Documents\PYTHON\MyDL\CNN\project_Seismic Signal Classification\data\example201.csv'
    modelpath = r'C:\Users\USER\Documents\PYTHON\MyDL\CNN\project_Seismic Signal Classification\data\xgbmodel_201.pkl'

    try:
        data = pd.read_csv(filepath, header=None, sep=',')
        data = data.dropna()
        
        x_test = data.iloc[:, 1:].values
        y_test = data.iloc[:, 0]

        model = joblib.load(modelpath)
        y_test_pred = model.predict(x_test)
        
        test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
        print('\n--- XGBoost Model 201 Sonuçları ---')
        print('XGBoost test Accuracy: {:.6f}'.format(test_accuracy))
        print('Misclassified samples: {}'.format((y_test != y_test_pred).sum()))
    except Exception as e:
        print(f"Model 201 xətası: {e}")

if __name__ == '__main__':
    xgbpredictone()
    xgbpredicttwo()