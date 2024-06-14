import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

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
    ],
        remainder = 'drop'
)

df = pd.read_csv('data/kc_house_data_NaN.csv').drop(columns='Unnamed: 0').dropna() 

                                                      
# X_train = df.drop(columns='price')
# y_train = df.price
# pipe.fit(X = X_train,y =  y_train)
# from sklearn.metrics import r2_score
# r2_score(y_train, y_pred)
LR = LinearRegression()
RFR = RandomForestRegressor()
XGB = XGBRegressor()
parameters = [ 
            {#baseline 
                'regressor': [LR],
                'regressor__fit_intercept': [True, False]
            },

            {
                'regressor': [RFR],
                'regressor__n_estimators': [100, 200, 500, 1000],
                'regressor__max_depth': [None, 10, 20, 30, 40, 50],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4, 10],
                'regressor__max_features': ['sqrt', 'log2'],
                'regressor__bootstrap': [True, False]
            },

            {
                'regressor': [XGB],
                'regressor__n_estimators': [100, 200, 500, 1000],
                'regressor__max_depth': [3, 4, 5, 6, 7, 8, 10],
                'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                'regressor__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'regressor__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'regressor__gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                'regressor__reg_alpha': [0, 0.01, 0.1, 1, 10],
                'regressor__reg_lambda': [0.1, 1, 10, 100]
            }
        ]

#evaluating multiple classifiers
#based on pipeline parameters
#-------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns='price'), df.price, test_size=0.20, random_state=42)

result=[]

for params in parameters:

    #classifier
    regressor = params['regressor'][0]
    print('running' , regressor)
    #getting arguments by
    #popping out regressor
    # print(params)
    params.pop('regressor')
    

    #pipeline
    steps=[('feature_creation_sqft_living', FeatureCreator_sqft('sqft_living',
                                                    'sqft_living15',
                                                    'change_sqft_living_flag')),
    ('feature_creation_sqft_lot', FeatureCreator_sqft('sqft_lot',
                                                    'sqft_lot15',
                                                    'change_sqft_lot_flag' )),
    ('feature_creation_age_of_property', FeatureCreator_age('yr_built','yr_renovated')),
    ('add_kmeans_feature', KMeansTransformer()),
    ('preprocessor', preprocessor),
    ('regressor', regressor)]

    
    #cross validation using
    #Grid Search
    grid = GridSearchCV(Pipeline(steps= steps),
                        param_grid=params,
                        cv=3,
                        scoring= ['neg_mean_squared_error'],
                        refit = 'neg_mean_squared_error',
                        n_jobs=-1
                             )
    
    grid.fit(X_train, y_train)
    # print(X_train)
    #storing result
    result.append(
        {
            'grid': grid,
            'regressor': grid.best_estimator_,
            'best score': grid.best_score_,
            'best params': grid.best_params_,
            'cv': grid.cv
        }
    )
    # print(result.grid, result.regressor, result.best_score,result.best_params)
    print('finished ', regressor)

#sorting result by best score
# result = sorted(result, key=itemgetter('best score'),reverse=True)

#saving best classifier
# grid = result[0]['grid']
# joblib.dump(grid, 'classifier.pickle')



# # Define the parameter grid
# param_grid = {
#                 # 'regressor': [RFR],
#                 'regressor__n_estimators': [100, 200, 500, 1000],
#                 'regressor__max_depth': [None, 10, 20, 30, 40, 50],
#                 'regressor__min_samples_split': [2, 5, 10],
#                 'regressor__min_samples_leaf': [1, 2, 4, 10],
#                 'regressor__max_features': ['auto', 'sqrt', 'log2'],
#                 'regressor__bootstrap': [True, False]
#             }

# # Initialize the XGBRegressor
# xgb = XGBRegressor()


#     #pipeline
# steps = [
#     ('feature_creation_sqft_living', FeatureCreator_sqft('sqft_living',
#                                                 'sqft_living15',
#                                                 'change_sqft_living_flag')),
# ('feature_creation_sqft_lot', FeatureCreator_sqft('sqft_lot',
#                                                 'sqft_lot15',
#                                                 'change_sqft_lot_flag' )),
# ('feature_creation_age_of_property', FeatureCreator_age('yr_built','yr_renovated')),
# ('add_kmeans_feature', KMeansTransformer()),
# ('preprocessor', preprocessor),
# ('regressor', RandomForestRegressor())
# ]
# # Initialize GridSearchCV
# grid_search = GridSearchCV(Pipeline(steps=steps), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
# # grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
# # Fit the GridSearchCV to find the best parameters
# grid_search.fit(X_train, y_train)
