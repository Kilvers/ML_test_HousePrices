import joblib
import pandas as pd
from preprocessors import *
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import argparse
import pickle

# #read in pipeline pickle file 
# regressor_filepath = 'models/regressor.pickle'
# X_test = pd.read_csv('data/X_test_data.csv')
# y_test = pd.read_csv('data/y_test_data.csv').drop(columns='Unnamed: 0')




# y_pred = pipeline.predict(X_test)

# pipeline.score(X_test, y_test)



# print(r2_score(y_test, y_pred))

# plt.plot(range(len(y_pred[:100])), y_pred[:100])
# plt.plot(range(len(y_pred[:100])), y_test[:100])


# import pandas as pd



def main(regressor_filepath, X_test_filepath, y_test_filepath=None):
    # Load the model
    pipeline = joblib.load(regressor_filepath)
    
    # Load the test data
    X_test = pd.read_csv(X_test_filepath)


    
    # make predicitons 
    y_pred = pipeline.predict(X_test)


    if y_test_filepath:
        y_test = pd.read_csv(y_test_filepath).drop(columns='Unnamed: 0')

        r2 = r2_score(y_test, y_pred)
        print(r2)

    else:
        
        #save predictions 
        predictions = pd.DataFrame(y_pred, columns=['Prediction'])
        predictions.to_csv("outputs/predictions", index=False)
        print("Predictions:", y_pred)
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the regressor with specified test data.')
    parser.add_argument('--regressor_filepath', type=str, required=True, help='Filepath to the saved regressor model.')
    parser.add_argument('--X_test_filepath', type=str, required=True, help='Filepath to the X test data CSV file.')
    parser.add_argument('--y_test_filepath', type=str, required=False, help='Filepath to the y test data CSV file.')
    
    args = parser.parse_args()

    main(args.regressor_filepath, args.X_test_filepath, args.y_test_filepath)
