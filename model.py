# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import requests
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Importing the dataset
df = pd.read_csv('feature-df.csv')
print(df.head())
''' PCA '''
def pc(cols,n):
    scaler = StandardScaler()
    scaler.fit(cols)
    cols = scaler.transform(cols)
    pca = PCA(n_components=n)
    pca.fit(cols)
    cols = pca.transform(cols)
    return cols
'''END PCA'''

def apply_PCA (df):
    ''' Defining columns for PCA '''
    midfield = ['Crossing', 'ShortPassing', 'LongPassing','Curve']
    df_pas = df[midfield]
    shooting = ['Finishing', 'HeadingAccuracy', 'Volleys','ShotPower','LongShots']
    df_shoot = df[shooting]
    skill = ['FKAccuracy', 'Volleys', 'Dribbling','Curve','BallControl','Balance','Skill Moves','Vision']
    df_skill = df[skill]
    defence = ['Interceptions', 'Positioning', 'Marking', 'StandingTackle', 'SlidingTackle']
    df_def = df[defence]
    mental = ['Aggression','Composure','Vision']
    df_ment = df[mental]
    physical = df[['Height','Weight','Acceleration','SprintSpeed','Agility','Reactions','Balance','Jumping','Stamina','Strength']]


    ''' Columns defind '''

    ''' Applying PCA to columns '''
    df_pas = pc(df_pas,2)
    df_shoot = pc(df_shoot,2)
    df_skill = pc(df_skill,2)
    df_def = pc(df_def,2)
    df_ment = pc(df_ment,1)
    physical = pc(physical,2)
    '''Applied'''

    ''' Append new columns with PCA to dataframe '''
    df['pass-1'] = df_pas.transpose()[0]
    df['pass-2'] = df_pas.transpose()[1]

    df['shot-1'] = df_shoot.transpose()[0]
    df['shot-2'] = df_shoot.transpose()[1]

    df['skill-1'] = df_skill.transpose()[0]
    df['skill-2'] = df_skill.transpose()[1]

    df['def-1'] = df_def.transpose()[0]
    df['def-2'] = df_def.transpose()[1]
    
    df['Physical-1'] = physical.transpose()[0]
    df['Physical-2'] = physical.transpose()[1]

    df['ment'] = df_ment
    ''' Appended '''

apply_PCA(df)
#Setting feature and predicted variables
features = ['Physical-1','Physical-2', 'pass-1', 'pass-2', 'shot-1', 'shot-2', 'skill-1',
       'skill-2', 'def-1', 'def-2', 'ment']

X = df[features]
y = df.Position
# Splitting the dataset into the Training set and Test set
train_X, val_X, train_y, val_y = train_test_split(X,y,test_size=0.2,random_state=0)

# Fitting Simple Linear Regression to the Training set
rf_mod = GradientBoostingClassifier(random_state=0)
rf_mod.fit(train_X,train_y)

# Predicting the Test set results
rf_pred = rf_mod.predict(val_X)

# Saving model to disk
pickle.dump(rf_mod, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))