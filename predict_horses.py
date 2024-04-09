# this program predicts and returns the horse that are predicted to win, place, and show with confidence scores
# the program uses the combined_data.csv file in the data folder
# the program uses the model file in the models folder
# if model file is not found, the program trains a new model and saves it in the models folder


# data format:
# rid,horseName,age,saddle,decimalPrice,isFav,trainerName,jockeyName,position,positionL,dist,weightSt,weightLb,overWeight,outHandicap,headGear,RPR,TR,OR,father,mother,gfather,runners,margin,weight,res_win,res_place

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import joblib


def inital_set_up():
    # get the current working directory
    current_dir = os.getcwd()

    # get the path to the data folder
    data_dir = os.path.join(current_dir, 'data')

    # get the path to the models folder
    models_dir = os.path.join(current_dir, 'models')

    # get the path to the combined data file
    combined_file = os.path.join(data_dir, 'combined_data.csv')

    # check if the combined data file exists
    if not os.path.exists(combined_file):
        print('Combined data file not found:', combined_file)
        print('Please run combine_data.py to create the combined data file')
        exit()
        
    # read the combined data file into a dataframe
    combined_df = pd.read_csv(combined_file)

    # check if the combined data file is empty
    if combined_df.empty:
        print('Combined data file is empty:', combined_file)
        print('Please check the combined data file')
        exit()
        
    # check if the models folder exists
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    # get the path to the model file
    model_file = os.path.join(models_dir, 'model.pkl')
    return combined_df, model_file

def load_model(combined_df, model_file):
    # check if the model file exists 
    if os.path.exists(model_file):
        # load the model from the file
        model = joblib.load(model_file)
        print('Model loaded from:', model_file)
        
        # create a StandardScaler object
        scaler = StandardScaler()
        
        # create a list of feature columns
        feature_cols = ['age', 'decimalPrice', 'isFav', 'RPR', 'TR', 'OR', 'runners', 'margin', 'weight', 'res_win', 'res_place']
        
        return model, scaler, feature_cols      
        
    else:
        
        print("Model file not found. Training a new model...")
        
        # mprevent could not convert string to float: 'nk' error
        combined_df = combined_df[combined_df['position'] != 'nk']
        combined_df = combined_df[combined_df['position'] != 'dist']
        combined_df = combined_df[combined_df['position'] != 'pu']
        combined_df = combined_df[combined_df['position'] != 'ur']
        combined_df = combined_df[combined_df['position'] != 'f']
        combined_df = combined_df[combined_df['position'] != 'ro']
        combined_df = combined_df[combined_df['position'] != 'su']
        
        # convert the position column to integer
        combined_df['position'] = combined_df['position'].astype(int)
        
        # create a new column to store the target variable
        combined_df['target'] = 0
        
        # set the target variable to 1 for horses that finished in the top 3 positions
        combined_df.loc[combined_df['position'] == 1, 'target'] = 1
        combined_df.loc[combined_df['position'] == 2, 'target'] = 2
        combined_df.loc[combined_df['position'] == 3, 'target'] = 3
        
        # create a list of feature columns
        feature_cols = ['age', 'decimalPrice', 'isFav', 'RPR', 'TR', 'OR', 'runners', 'margin', 'weight', 'res_win', 'res_place']
        
        # create the feature matrix
        X = combined_df[feature_cols]
        
        # create the target variable
        y = combined_df['target']
        
        # split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        # create a StandardScaler object
        scaler = StandardScaler()
        
        # fit the scaler on the training data
        scaler.fit(X_train)
        
        # transform the training and testing data
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # create a RandomForestClassifier object
        model = RandomForestClassifier(n_estimators=100, random_state=0)
        
        # train the model on the training data
        model.fit(X_train_scaled, y_train)
        
        # make predictions on the testing data
        y_pred = model.predict(X_test_scaled)
        
        # calculate the accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)
        
        # calculate the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        print('Model trained successfully')
        
        print('Classification Report:')
        
        print('Accuracy:', accuracy)
        print('Confusion Matrix:')
        print(cm)
        
        # save the model to a file
        joblib.dump(model, model_file)
        print('Model saved to:', model_file)
        
        
        return model, scaler, feature_cols
  
  
def predict_horses(combined_df, model, scaler, feature_cols):
    # get the list of unique horse names
    horse_names = combined_df['horseName'].unique()
    
    # create an empty dataframe to store the predictions
    predictions_df = pd.DataFrame(columns=['horseName', 'win', 'place', 'show'])
    
    # loop through the list of horse names
    for horse_name in horse_names:
        # get the data for the horse
        horse_data = combined_df[combined_df['horseName'] == horse_name]
        
        # get the latest data for the horse
        horse_data = horse_data.tail(1)
        
        # create the feature matrix for the horse
        X = horse_data[feature_cols]
        
        # transform the data using the scaler
        X_scaled = scaler.transform(X)
        
        # make predictions using the model
        y_pred = model.predict_proba(X_scaled)
        
        # get the confidence scores for the predictions
        win_score = y_pred[0][1]
        place_score = y_pred[0][2]
        show_score = y_pred[0][3]
        
        # add the predictions to the dataframe
        predictions_df = predictions_df.append({'horseName': horse_name, 'win': win_score, 'place': place_score, 'show': show_score}, ignore_index=True)
        
    # sort the predictions by the win score
    predictions_df = predictions_df.sort_values(by='win', ascending=False)
    
    
    

# main function
def main():
    # set up the initial variables
    combined_df, model_file = inital_set_up()
    
    # load the model
    model, scaler, feature_cols = load_model(combined_df, model_file)
    
    # predict the horses
    predict_horses(combined_df, model, scaler, feature_cols)
    
    # print the predictions
    print(predictions_df.head(10))
    
    # save the predictions to a file
    predictions_file = os.path.join(data_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_file, index=False)
    print('Predictions saved to:', predictions_file)
    
    

if __name__ == '__main__':
    main()