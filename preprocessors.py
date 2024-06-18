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



# Custom transformer to create new features
class FeatureCreator_sqft(BaseEstimator, TransformerMixin):
    """
        Custom transformer for creating new features related to square footage.

        This transformer calculates the difference between two specified square footage columns
        and generates a new feature indicating whether the difference is positive, negative, or zero.

        Parameters
        ----------
        sqrt : str
            The column name for the square footage of the house.
        sqft15 : str
            The column name for the square footage of the 15 nearest neighbors.
        sqrt_flag : str
            The column name for the flag indicating whether the difference between `sqft15` and `sqrt` is positive, negative, or zero.
    """
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
    """
        Custom transformer for creating new features related to the age of the property and the years since last renovation.

        This transformer calculates the age of the property and the years since the last renovation
        based on the given year built and year renovated columns.

        Parameters
        ----------
        yr_built : str
            The column name for the year the property was built.
        yr_reno : str
            The column name for the year the property was last renovated.
    """
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
    """
        Custom transformer for applying KMeans clustering to geographical coordinates.

        This transformer uses the KMeans algorithm to cluster data points based on their latitude and longitude
        and adds the cluster labels as a new feature to the dataframe.

        Parameters
        ----------
        n_clusters : int, optional (default=15)
            The number of clusters to form as well as the number of centroids to generate.
        """
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