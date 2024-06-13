import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

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
    def __init__(self): #, clusters=15
        self.kmeans = KMeans(n_clusters=15,max_iter=1000, random_state= 42)
        self.labels_ = None 

    def fit(self, X, y=None):
        self.kmeans.fit(X[['lat', 'long']])
        return self

    def transform(self, X):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Add cluster labels to the DataFrame
        # print(self.tablbe)
        X['cluster'] = self.kmeans.labels_
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
                         'cluster']
categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

df = pd.read_csv('data/kc_house_data_NaN.csv').drop(columns='Unnamed: 0').dropna() 


pipe = Pipeline(  steps=[
    ('feature_creation_sqft_living', FeatureCreator_sqft('sqft_living',
                                                    'sqft_living15',
                                                    'change_sqft_living_flag')),
    ('feature_creation_sqft_lot', FeatureCreator_sqft('sqft_lot',
                                                    'sqft_lot15',
                                                    'change_sqft_lot_flag' )),
    ('feature_creation_age_of_property', FeatureCreator_age('yr_built','yr_renovated')),
    ('add_kmeans_feature', KMeansTransformer()),
    ('preprocessor', preprocessor),
    ('clf', LinearRegression())

                               

                                                                   ])


X_train = df.drop(columns='price')
y_train = df.price
pipe.fit(X = X_train,y =  y_train)
from sklearn.metrics import r2_score
r2_score(y_train, y_pred)
