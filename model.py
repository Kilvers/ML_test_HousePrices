import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from operator import itemgetter
import joblib


import preprocessors


if __name__ == "__main__":
    
    df = pd.read_csv('data/kc_house_data_NaN.csv').drop(columns='Unnamed: 0').dropna() 


    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns='price'), df.price.values, test_size=0.20, random_state=42)

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
                    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3], #"eta "Step size shrinkage used in update to prevents overfitting. 
                    'regressor__subsample': [0.6, 0.7, 0.8, 0.9, 1.0], # amount of shuffling
                    # 'regressor__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0], #the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
                    # 'regressor__gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5], #Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.
                    # 'regressor__reg_alpha': [0, 0.01, 0.1, 1, 10], #L1 regularization term on weights. Increasing this value will make model more conservative.
                    'regressor__reg_lambda': [0.1, 1, 10, 100] #L2 regularization term on weights. Increasing this value will make model more conservative
                }
            ]

    #evaluating multiple classifiers
    #based on pipeline parameters
    #-------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns='price'), df.price.values, test_size=0.20, random_state=42)

    result=[]

    for params in parameters:

        #classifier
        regressor = params['regressor'][0]
        print('running' , regressor)


        #popping out regressor as we don't want it as a param
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

    for i in result:
        print('bestscore =  ', i['best score'])
        print(i['regressor'].score(X_test,y_test))
        print('best_params = ', i["best params"] )

    #sorting result by best score

    result = sorted(result, key=itemgetter('best score'),reverse=True)

    # saving best regressor
    grid = result[0]['grid']
    joblib.dump(grid, 'models/regressor.pickle')
    joblib.dump(grid.best_estimator_.named_steps['add_kmeans_feature'].kmeans, "models/kmeans.pickle")

    X_test.to_csv('data/X_test_data.csv')
    pd.DataFrame(y_test,columns=['y']).to_csv('data/y_test_data.csv')