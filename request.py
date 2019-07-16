import requests
from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

# Load the model
model = pickle.load(open('model.pkl','rb'))

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
    print(df.columns)
    ''' Defining columns for PCA '''
    midfield = ['Crossing', 'ShortPassing', 'LongPassing','Curve']
    df_pas = df[midfield]
    shooting = ['Finishing', 'HeadingAccuracy', 'Volleys','ShotPower','LongShots']
    df_shoot = df[shooting]
    skill = ['FKAccuracy', 'Volleys', 'Dribbling','Curve','BallControl','Balance','Skill Moves','Vision', 'Penalties']
    df_skill = df[skill]
    defence = ['Interceptions', 'Positioning', 'Marking', 'StandingTackle', 'SlidingTackle']
    df_def = df[defence]
    mental = ['Aggression','Composure','Vision']
    df_ment = df[mental]
    physical=df[['Height','Weight','Acceleration','SprintSpeed','Agility','Reactions','Balance','Jumping','Stamina','Strength']]


    ''' Columns defind '''

    ''' Applying PCA to columns '''
    print(df_pas)
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

def to_pca(row):
    df = pd.read_csv('feature-df.csv')
    features = ['Height','Weight','Crossing','Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle','Skill Moves']
    df = df[features]
    print(df.shape)
    print(len(row))
    
    #df = pd.concat([df, row], axis=0)
    df.loc[len(df.index)]=row
    print(df.shape)
    #df = df.drop([df.columns[len(df.columns)-1]], axis=1, inplace=True)
    print(df.tail())
    apply_PCA(df)
    print(df.iloc[[-1]])
    return df.iloc[[-1]]

def to_position(pred_arr):
    i=0
    Y = None
    x = pred_arr
    if x==1:
        Y="Striker"
    elif x==3:
        Y="Winger"
    elif x==4:
        Y="Keeper"
    elif x==5:
        Y="Center-mid"
    elif x==8:
        Y="Center-back"
    elif x==11:
        Y="Holding-mid"
    elif x==12:
        Y="Attacking-mid"
    elif x==16:
        Y="Wide-mid"
    elif x==19:
        Y="Full-back"
    elif x==26:
        Y="Wing-back"
    i = i+1
    return Y

new_arr = [72,20,70,60,55,65,60,70,65,60,60,65,70,70,70,50,70,60,55,55,60,65,50,35,40,59,60,65,36,30,28,4]
new_vals = pd.DataFrame(columns=['Index'])
new_vals = new_vals.append([new_arr])

url = 'http://localhost:5000/api'
print(new_arr)

app = Flask(__name__)
i=0
features = [None]*32
FEATURES = ['Height','Weight','Crossing','Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
'Marking', 'StandingTackle', 'SlidingTackle','Skill Moves']

@app.route('/', methods=['GET','POST'])
def send():
    global i, features, FEATURES

    if request.method == 'POST':
        features[i] = request.form[str(i)]
        i += 1
        if features[31] != None:
            ftr = [x for x in features]
            r = requests.post(url,json={'exp':ftr,})
            pred = to_position(r.json())  
            i = 0
            return render_template('result.html', position=pred)
        else:
            return render_template('index.html', i=i, features=FEATURES)

        
    else:
        return render_template('index.html', i=i, features=FEATURES)
    
@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    
    new_vals = pd.DataFrame(columns=['Index'])
    new_vals = new_vals.append(data['exp'])
    new_vals = new_vals.drop('Index',axis=1)
    print(new_vals.shape)
    
    row = np.asarray(data['exp'])
    #row = np.pad(row, (0,1), 'constant')
    row = to_pca(data['exp'])
    features = ['Physical-1','Physical-2', 'pass-1', 'pass-2', 'shot-1', 'shot-2', 'skill-1',
       'skill-2', 'def-1', 'def-2', 'ment']
    row = row[features]
    #prediction = model.predict(np.asarray(data['exp']).reshape((1, -1)))
    prediction = model.predict(row)
    # Take the first value of prediction
    output = int(prediction)
    return jsonify(output)

if __name__ == "__main__":
    app.run()


#Decode position variable
