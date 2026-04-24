import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn import metrics

def train_and_save_svm(csv_path, model_save_path):
    print(f"Hazırlanır: {csv_path}")
    
    try:
        data = pd.read_csv(csv_path, header=None, sep=',')
        data = data.dropna()
        
        X = data.iloc[:, 1:].values
        y = data.iloc[:, 0].values
        
        model = SVC(kernel='rbf', probability=True) 
        model.fit(X, y)
        
        joblib.dump(model, model_save_path)
        
        y_pred = model.predict(X)
        accuracy = metrics.accuracy_score(y, y_pred)
        
        print(f"Uğurlu! Model yadda saxlanıldı: {model_save_path}")
        print(f"Train Accuracy: {accuracy:.6f}\n")
        
    except Exception as e:
        print(f"Xəta baş verdi ({csv_path}): {e}")

if __name__ == '__main__':
    path_36_csv = r'C:\Users\USER\Documents\PYTHON\MyDL\CNN\project_Seismic Signal Classification\data\example36.csv'
    path_36_model = r'C:\Users\USER\Documents\PYTHON\MyDL\CNN\project_Seismic Signal Classification\data\svmmodel_36.pkl'
    train_and_save_svm(path_36_csv, path_36_model)

    path_201_csv = r'C:\Users\USER\Documents\PYTHON\MyDL\CNN\project_Seismic Signal Classification\data\example201.csv'
    path_201_model = r'C:\Users\USER\Documents\PYTHON\MyDL\CNN\project_Seismic Signal Classification\data\svmmodel_201.pkl'
    train_and_save_svm(path_201_csv, path_201_model)