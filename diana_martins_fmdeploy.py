#imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from xgboost.sklearn import XGBClassifier
from pydub import AudioSegment
import os
import librosa
import plotly.graph_objs as go
import glob

#PATHS
PATH_CSV = './features.csv'
#what is the file bellow? where is it created? does this code only work for classical sounds?
#is this the ouput file?
PATH_CSV_CLASSICAL_TIMESTAMPS = './features_classical_timestamps.csv'
#directory for the audio chunks .wav
PATH_CLASSICAL_TIMESTAMPS = './classical_timestamps'
#what are these for? defined but not used
CLASSES_NAMES=['negative', 'positive', 'unknown']
#input file
audio_file= "./classical.00010.wav"

#read features.csv
#is this file allways the same?
data=pd.read_csv(PATH_CSV)

#new column with values bellow
def new_column(csv_file):
    label_2 = pd.Series([]) 
  
    for i in range(len(data)): 
        if data['label'][i] == 0: 
            label_2[i]='negative'
  
        elif data['label'][i] == 7: 
            label_2[i]='unknown'
  
        elif data['label'][i] == 10: 
            label_2[i]='unknown'
        
        elif data['label'][i] == 11: 
            label_2[i]='unknown'
    
        elif data['label'][i] == 13: 
            label_2[i]='unknown'
        
        elif data['label'][i] == 14: 
            label_2[i]='unknown'
    
        elif data['label'][i] == 16: 
            label_2[i]='unknown'
        
        elif data['label'][i] == 20: 
            label_2[i]='negative'
        
        elif data['label'][i] == 36: 
            label_2[i]='negative'
    
        elif data['label'][i] == 42: 
            label_2[i]='negative'
        
        elif data['label'][i] == 48: 
            label_2[i]='negative'
        
        elif data['label'][i] == 51: 
            label_2[i]='positive'
        
        elif data['label'][i] == 52: 
            label_2[i]='positive'
  
        else: 
            label_2[i]= data['label'][i] 
  
    # inserting new column with values of list made above         
    data.insert(2, 'label_2', label_2) 
    
new_column(data)

def load_data(data):  
    X = data.drop([data.columns[0],'filename', 'label', 'label_2'],axis=1) #droping first column, filename column and label column
    Y= data['label_2'] 
    print('X:', X.shape)
    print('Y:', Y.shape)
    return (X, Y)

(X, Y) = load_data(data)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)

enc = OneHotEncoder(handle_unknown='ignore')
Y_train = enc.fit_transform(y_train.values.reshape(-1,1)).toarray()
Y_test = enc.fit_transform(y_test.values.reshape(-1,1)).toarray()
train = np.argmax(Y_train,axis =  1)
test = np.argmax(Y_test, axis = 1)

clf = XGBClassifier(learning_rate=0.1,
                    n_estimators=400,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='multi:softprob',
                    nthread=4,
                    num_class=3,
                    seed=27,
                    predictor = 'gpu_predictor',
                    tree_method='gpu_hist',
                    gpu_id=0)


clf.fit(X_train,train.ravel(),
            verbose=False,
            early_stopping_rounds=50,
            eval_metric='merror',
            eval_set=[(X_test, test.ravel())])

y_predict = clf.predict(X_test)
y_train_predict = clf.predict(X_train)

os.makedirs('./classical_timestamps', exist_ok=True) #wont raise OSError if directory already exists

audio = AudioSegment.from_wav(audio_file)
#does this work if the file is > 30?
#what happens if the .wav is more then 30s?
list_of_timestamps = [5, 10, 15, 20, 25, 30] #and so on in *seconds*

#create audio chunks in intervals of 5
start = 0
for  idx,t in enumerate(list_of_timestamps):
    #break loop if at last element of list
    if idx == len(list_of_timestamps):
        break

    end = t * 1000 #pydub works in millisec
    audio_chunk=audio[start:end]
    audio_chunk.export( "./classical_timestamps/classical_{}.wav".format(end), format="wav") 

    start = end  #pydub works in millisec

#csv
def create_csv_header():
    header = ['filename','label','chroma_stft','mel_spectogram','spectral_contrast','tonnetz']
    for i in range(1, 41):
        header.append(f' mfcc{i}')
    return header

header = create_csv_header()

def generate_csv_music(PATH_CLASSICAL_TIMESTAMPS, path_csv, file_ext='*.wav'):
    if os.path.exists(path_csv):
        print(f'CSV {path_csv} already exists')
    else:
        sound_cases=[]
        for fn in glob.glob(os.path.join(PATH_CLASSICAL_TIMESTAMPS, file_ext)): #classical music
            filename = fn.split('\\')[1]
            label = 3
            sound_data=[filename]
            sound_data.append(label)
            sound = f'{PATH_CLASSICAL_TIMESTAMPS}/{filename}'
            X, sr = librosa.load(sound)
            stft = np.abs(librosa.stft(X))
            sound_data.append(np.mean(librosa.feature.chroma_stft(S=stft, sr=sr)))
            sound_data.append(np.mean(librosa.feature.melspectrogram(X, sr=sr)))
            sound_data.append(np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr)))
            sound_data.append(np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr)))
            mfcc = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40)
            for e in mfcc:
                sound_data.append(np.mean(e))
            sound_cases.append(sound_data)
                
        df = pd.DataFrame(sound_cases, columns = header) 
        df.to_csv(path_csv)
        
generate_csv_music(PATH_CLASSICAL_TIMESTAMPS, PATH_CSV_CLASSICAL_TIMESTAMPS)

DATA_3=pd.read_csv(PATH_CSV_CLASSICAL_TIMESTAMPS)

def load_data(data):  
    X_TEST_CLASSICAL = data.drop([data.columns[0],'filename', 'label'],axis=1)#droping first column, filename column and label column 
    Y_TEST_CLASSICAL = data['label']
    print('X_TEST_CLASSICAL:', X_TEST_CLASSICAL.shape)
    print('Y_TEST_CLASSICAL:', Y_TEST_CLASSICAL.shape)
    return (X_TEST_CLASSICAL, Y_TEST_CLASSICAL)

(X_TEST_CLASSICAL,Y_TEST_CLASSICAL) = load_data(DATA_3)

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return indices

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

def print_series_prediction_3(sound, label):
    sounds = []
    audio = []
    first_preds = []
    first_percent = []
    second_preds = []
    second_percent = []
    third_preds = []
    third_percent = []
    for i in range(len(sound)): #para imprimir tabela de previsoes
        predictions = clf.predict_proba(sound)
        predic = largest_indices(predictions[i], 3)
        percentage = predictions[i][predic]
        
        sounds.append(label[i])
        #audio.append(display(Audio(AUDIO, autoplay=True, rate=rate)))
        first_preds.append((predic)[0])
        first_percent.append(truncate(percentage[0],3))
        second_preds.append((predic)[1])
        second_percent.append(truncate(percentage[1],3))
        third_preds.append((predic)[2])
        third_percent.append(truncate(percentage[2],3))
        mean_first = sum(first_preds)/ len(first_preds)
        mean_first = int(round(mean_first))
        mean_second = sum(second_preds)/ len(second_preds)
        mean_second = int(round(mean_second))
        mean_third = sum(third_preds)/ len(third_preds)
        mean_third = int(round(mean_third))
       
    data = {'Y_test':  sounds, #'Sound': audio, 'mean_first' : mean_first, 'mean_second' : mean_second,'mean_third': mean_third 
            'first_pred': first_preds,
            'first_percent': first_percent,
            'second_pred': second_preds,
            'second_percent': second_percent,
            'third_pred': third_preds,
            'third_percent': third_percent
           }
    
    df = pd.DataFrame(data)
    fig = go.Figure(data=[go.Table(
    header=dict(values=list(df.columns),
                fill_color='lavender',
                align='center'),
    cells=dict(values=[df.Y_test, df.first_pred, df.first_percent, df.second_pred, # df.Sound,df.mean_first,df.mean_second,df.mean_third
                       df.second_percent, df.third_pred, df.third_percent],
               #fill_color='lightgrey',
               fill=dict(color=['lightgrey', 'white']),
               align='center'))])

    fig.show()
    
    print('Mean of timestamps:', mean_first)
    
    return df

df = print_series_prediction_3(X_TEST_CLASSICAL, Y_TEST_CLASSICAL)