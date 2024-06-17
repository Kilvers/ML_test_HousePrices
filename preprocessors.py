# import model 
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from operator import itemgetter
import joblib


regressor_filepath = 'models/regressor.pickle'
X_test = pd.read_csv('data/X_test_data.csv')
y_test = pd.read_csv('data/y_test_data.csv')

# Custom transformer to create new features
class FeatureCreator_sqft(BaseEstimator, TransformerMixin):
    def __init__(self, sqrt, sqft15, sqrt_flag):
        self.sqrt = sqrt
        self.sqft15 = sqft15
        self.sqrt_flag = sqrt_flag

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['temp'] = X[self.sqft15] - X[self.sqrt]
        X[self.sqrt_flag] = X['temp'].apply(lambda x: -1 if x < 0 else (1 if x > 0 else 0))
        # X = X.drop(columns = 'temp')"?"
        return X
    
class FeatureCreator_age(BaseEstimator, TransformerMixin):
    def __init__(self, yr_built, yr_reno ):
        self.yr_built = yr_built
        self.yr_reno = yr_reno

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['age_of_property'] = 2016 - X[self.yr_built ]
        X['yr_since_last_reno'] = 2016 - X[self.yr_reno ]
        return X

    
class KMeansTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=15):
        self.n_clusters = n_clusters
        self.kmeans = None


    def fit(self, X, y=None):
        # Fit KMeans using the specified columns
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.kmeans.fit(X[['lat', 'long']])
        return self

    def transform(self, X):
        # Predict cluster labels
        clusters = self.kmeans.predict(X[['lat', 'long']])
        # Add cluster labels as a new column to the DataFrame
        X = X.copy()  # Avoid modifying the original DataFrame
        X['cluster'] = clusters
        return X


numerical_features = ['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors','view', 'condition', 'grade',
       'sqft_above', 'sqft_basement','yr_since_last_reno','age_of_property']
numerical_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_features = ['waterfront',
                         'view',
                         'change_sqft_living_flag',
                         'change_sqft_lot_flag',
                         'cluster'
                         ]
categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ],
        remainder = 'drop'
)