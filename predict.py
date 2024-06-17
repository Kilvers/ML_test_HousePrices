# import model 
import joblib
import pandas as pd
import preprocessors
import joblib
from sklearn.metrics import r2_score

regressor_filepath = 'models/regressor.pickle'
X_test = pd.read_csv('data/X_test_data.csv')
y_test = pd.read_csv('data/y_test_data.csv').drop(columns='Unnamed: 0')


pipeline = joblib.load(regressor_filepath)

y_pred = pipeline.predict(X_test)

pipeline.score(X_test, y_test)



r2_score(y_test, y_pred)