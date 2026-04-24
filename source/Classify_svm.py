import numpy as np
import pandas as pd
from sklearn import metrics
import joblib

def svmpredictone():
    # 'r' prefixi mütləqdir
    filepath = r'C:\Users\USER\Documents\PYTHON\MyDL\CNN\project_Seismic Signal Classification\data\example36.csv'
    modelpath = r'C:\Users\USER\Documents\PYTHON\MyDL\CNN\project_Seismic Signal Classification\data\svmmodel_36.pkl'

    try:
        data = pd.read_csv(filepath, header=None, sep=',')
        data = data.dropna()
        x_test = data.iloc[:, 1:]
        y_test = data.iloc[:, 0]

        model = joblib.load(modelpath)
        y_test_pred = model.predict(x_test)
        
        test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
        print('--- SVM Model 36 Nəticələri ---')
        print('SVM test Accuracy: {:.6f}'.format(test_accuracy))
        print('Misclassified samples: {}'.format((y_test != y_test_pred).sum()))
    except Exception as e:
        print(f"Model 1 xətası: {e}")

def svmpredicttwo():
    # Burada düzəliş edildi: data faylı .csv olmalıdır, model faylı .pkl
    filepath = r'C:\Users\USER\Documents\PYTHON\MyDL\CNN\project_Seismic Signal Classification\data\example201.csv' 
    modelpath = r'C:\Users\USER\Documents\PYTHON\MyDL\CNN\project_Seismic Signal Classification\data\svmmodel_201.pkl'

    try:
        data = pd.read_csv(filepath, header=None, sep=',')
        data = data.dropna()
        x_test = data.iloc[:, 1:]
        y_test = data.iloc[:, 0]

        model = joblib.load(modelpath)
        y_test_pred = model.predict(x_test)
        
        test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
        print('\n--- SVM Model 201 Nəticələri ---')
        print('SVM test Accuracy: {:.6f}'.format(test_accuracy))
        print('Misclassified samples: {}'.format((y_test != y_test_pred).sum()))
    except Exception as e:
        print(f"Model 2 xətası: {e}")

if __name__ == '__main__':
    svmpredictone()
    svmpredicttwo()