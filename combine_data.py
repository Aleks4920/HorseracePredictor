# this program combines all the csv files from the archive folder into one csv file
# the combined csv file is saved in the data folder
# the combined csv file is named combined_data.csv

import os
import pandas as pd

# get the current working directory
current_dir = os.getcwd()

# get the path to the archive folder
archive_dir = os.path.join(current_dir, 'archive')

# get the path to the data folder
data_dir = os.path.join(current_dir, 'data')

# get the list of files in the archive folder
files = os.listdir(archive_dir)

# create an empty list to store the dataframes
dfs = []

# loop through the files in the archive folder
for file in files:
    # get the path to the file
    file_path = os.path.join(archive_dir, file)
    
    # read the file into a dataframe
    df = pd.read_csv(file_path)
    
    # append the dataframe to the list
    dfs.append(df)
    
# concatenate the dataframes in the list
combined_df = pd.concat(dfs)

# save the combined dataframe to a csv file
combined_file = os.path.join(data_dir, 'combined_data.csv')
combined_df.to_csv(combined_file, index=False)

print('Combined data saved to:', combined_file)
print('Number of rows in combined data:', combined_df.shape[0])
print('Number of columns in combined data:', combined_df.shape[1])