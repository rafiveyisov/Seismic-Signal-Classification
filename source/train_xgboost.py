import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn import metrics
import warnings

warnings.filterwarnings("ignore")

def train_and_save_xgb(csv_path, model_save_path):
    print(f"XGBoost öyrədilir: {csv_path}")
    
    try:
        data = pd.read_csv(csv_path, header=None, sep=',')
        data = data.dropna()
        
        X = data.iloc[:, 1:].values
        y = data.iloc[:, 0].values
        
        model = XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            use_label_encoder=False, 
            eval_metric='logloss'
        )
        
        model.fit(X, y)
        
        joblib.dump(model, model_save_path)
        
        y_pred = model.predict(X)
        accuracy = metrics.accuracy_score(y, y_pred)
        
        print(f"Uğurlu! XGBoost modeli yeniləndi: {model_save_path}")
        print(f"Train Accuracy: {accuracy:.6f}\n")
        
    except Exception as e:
        print(f"Xəta baş verdi ({csv_path}): {e}")

if __name__ == '__main__':
    path_36_csv = r'C:\Users\USER\Documents\PYTHON\MyDL\CNN\project_Seismic Signal Classification\data\example36.csv'
    path_36_model = r'C:\Users\USER\Documents\PYTHON\MyDL\CNN\project_Seismic Signal Classification\data\xgbmodel_36.pkl'
    train_and_save_xgb(path_36_csv, path_36_model)

    path_201_csv = r'C:\Users\USER\Documents\PYTHON\MyDL\CNN\project_Seismic Signal Classification\data\example201.csv'
    path_201_model = r'C:\Users\USER\Documents\PYTHON\MyDL\CNN\project_Seismic Signal Classification\data\xgbmodel_201.pkl'
    train_and_save_xgb(path_201_csv, path_201_model)