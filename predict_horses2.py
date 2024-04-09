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
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer


models_dir = os.path.join(os.getcwd(), 'models')



def inital_set_up():
    # get the current working directory
    current_dir = os.getcwd()
    
    # get the path to the data folder
    data_dir = os.path.join(current_dir, 'data')
    
    # get the path to the models folder
    models_dir = os.path.join(current_dir, 'models')
    
    # get the path to the combined data file
    combined_file = os.path.join(data_dir, 'combined_data_test.csv')
    
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
    scaler_file = os.path.join(models_dir, 'scaler.pkl')    
    
    return combined_df, model_file, scaler_file


def create_model(combined_df, models_dir='models'):
    print("Model file not found. Training a new model...")
    
    # Filtering out specific positions
    positions_to_remove = ['nk', 'dist', 'pu', 'ur', 'f', 'ro', 'su']
    combined_df = combined_df[~combined_df['position'].isin(positions_to_remove)]
    
    combined_df['position'] = combined_df['position'].astype(int)
    
    
    # Assuming 'position' is the target and rest are features
    feature_cols = ['age', 'decimalPrice', 'isFav', 'RPR', 'TR', 'OR', 'runners', 'margin', 'weight']
    X = combined_df[feature_cols]
    y = combined_df['position']
    
    
    
    # Ensure there's no class with only 1 member
    class_counts = y.value_counts()
    min_class_count = class_counts.min()
    if min_class_count < 2:
        # Handling classes with less than 2 instances
        y = y[y.isin(class_counts[class_counts >= 2].index)]
        X = X.loc[y.index]
    
    # Continue with your existing setup
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    selector = SelectKBest(f_classif, k=9)
    X_train_scaled = selector.fit_transform(X_train_scaled, y_train)
    
    # Adjusting n_splits based on the smallest class size
    n_splits = min(class_counts[class_counts >= 2].min(), 5)  # Ensure at least 2 instances per class, use 5 or fewer splits
    cv_strategy = StratifiedKFold(n_splits=n_splits)
    
    # Grid Search for Hyperparameter Tuning
    # param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
    # grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=cv_strategy, scoring='accuracy')
    # grid_search.fit(X_train_scaled, y_train)
    # print('Best parameters:', grid_search.best_params_)
    # print('Best score:', grid_search.best_score_)
    
    # Selected features based on importance
    model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Cross-validation with the adjusted strategy
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_strategy)
    print("Cross-validation scores:", scores)
    
    # Model evaluation
    y_pred = model.predict(X_test_scaled)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    
    # Save model and scaler
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    joblib.dump(model, os.path.join(models_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    
    return model, scaler, feature_cols





def load_model(model_file, scaler_file, combined_df):
    if os.path.exists(model_file):
        model = joblib.load(model_file)
        print('Model loaded from:', model_file)
    else:
        print("Model file not found.")
        create_model(combined_df)
        model = joblib.load(model_file)
        print('Model loaded from:', model_file)

    if os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
        print('Scaler loaded from:', scaler_file)
    else:
        print("Scaler file not found.")
        create_model(combined_df)
        scaler = joblib.load(scaler_file)
        print('Scaler loaded from:', scaler_file)
        
    # Assuming feature_cols remains defined in this scope
    feature_cols = ['age', 'decimalPrice', 'isFav', 'RPR', 'TR', 'OR', 'runners', 'margin', 'weight']
    return model, scaler, feature_cols

def predict_horses(model, scaler, feature_cols, horse_names, combined_df):
    # look at all the horses data and predict the positions of the horses and return the predictions with cofidence percentage and precetage of the horses winning 1st place
    for horse_name in horse_names:
        # get the horse data from the combined dataframe
        horse_data = get_horse_data(combined_df, horse_name)
        
        # check if the horse data is empty
        if horse_data is None:
            continue
        
        # get the features for the horse
        horse_features = horse_data[feature_cols]
        
        # transform the horse features
        horse_features_scaled = scaler.transform(horse_features)
        
        # predict the horse position
        position = model.predict(horse_features_scaled)[0]
        
        # predict the horse position probabilities
        position_probs = model.predict_proba(horse_features_scaled)[0]
        
        # get the probability of the horse winning
        win_prob = position_probs[0]
        
        # print the horse name, position, and win probability
        print('Horse:', horse_name)
        print('Predicted Position:', position)
        print('Win Probability:', win_prob)
        print('-------------------------')

def predict_and_rank_horses(model, scaler, feature_cols, race_data):

    # Predicts and ranks horses in a given race.
    
    # store the predictions
    predictions = []
    
    # loop through the horse data
    for horse in race_data:
        
        horse_name = horse['horseName'].values[0]
        
        # get the features for the horse
        horse_features = horse[feature_cols]
        
        
        
        # transform the horse features
        horse_features_scaled = scaler.transform(horse_features)
        
        # predict the horse position
        position = model.predict(horse_features_scaled)[0]
        
        # predict the horse position probabilities
        position_probs = model.predict_proba(horse_features_scaled)[0]
        
        # get the probability of the horse winning
        win_prob = position_probs[0]
        
        # store the horse name, position, and win probability and confidence
        predictions.append((horse_name, position, win_prob, position_probs[position-1]))
        
    # sort the predictions by win probability
    predictions.sort(key=lambda x: x[2], reverse=True)
    
    # rank the horses if two horses have the same win probability the horse with the higher position probability is ranked higher
    # if those are the same then the first horse is ranked higher
    rank = 1
    for i, prediction in enumerate(predictions):
        print('Rank:', rank)
        print('Horse:', prediction[0])
        print('Predicted Position:', prediction[1])
        print('Win Probability:', prediction[2])
        print('Position Probability:', prediction[3])
        print('-------------------------')
        rank += 1
        
    return predictions



# lookup the horse data from the combined dataframe
def get_horse_data(combined_df, horse_name):
    # check if the combined dataframe is empty
    if combined_df.empty:
        print('Combined dataframe is empty')
        return None

    # check if the horse name is in the combined dataframe
    if horse_name not in combined_df['horseName'].values:
        print('Horse not found:', horse_name)
        return None

    # get the horse data from the combined dataframe
    horse_data = combined_df[combined_df['horseName'] == horse_name]
    return horse_data

def main():
    # use a list of horse names to test the function
    horse_names = ['Combermere', 'Bickfield', 'Self Aid', 'Carfax', 'Cima', 'Father Pat', 'Hot Hope']
    
    # get the combined dataframe, model file, and scaler file
    combined_df, model_file, scaler_file = inital_set_up()
    
    # load the model and scaler
    model, scaler, feature_cols = load_model(model_file, scaler_file, combined_df)
    
    # predict the horses
    predict_horses(model, scaler, feature_cols, horse_names, combined_df)
    
    # predict the race outcome
    
    # Create a list of horse data for
    # the horses
    horse_data = []
    
    # look up the horse data from the combined dataframe
    horse_data.append(get_horse_data(combined_df, 'Combermere'))
    horse_data.append(get_horse_data(combined_df, 'Bickfield'))
    horse_data.append(get_horse_data(combined_df, 'Self Aid'))
    horse_data.append(get_horse_data(combined_df, 'Carfax'))
    horse_data.append(get_horse_data(combined_df, 'Cima'))
    horse_data.append(get_horse_data(combined_df, 'Father Pat'))
    
    # predict and rank the horses
    predict_and_rank_horses(model, scaler, feature_cols, horse_data)
    
        
if __name__ == '__main__':
    main()
    