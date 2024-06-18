# ML_test_HousePrices
developed using python 3.12     
### to retrain the pipeline
 you can run model.py from the command line   
 then install requirement.txt using  
 - pip install -r requirements.txt  
    nb : recommended  create a new virtual environment 
    
### to score the model use: 
- python predict.py --regressor_filepath models/regressor.pickle --X_test_filepath data/X_test_data.csv --y_test_filepath data/y_test_data.csv

### to predict new data use without y values 
 - python predict.py --regressor_filepath models/regressor.pickle --X_test_filepath data/X_test_data.csv
  
 Currently, this then saves the outputs to outputs/predictions


