#imports
import os
import pandas as pd
import glob
import numpy as np
import librosa
import plotly.graph_objs as go
from xgboost.sklearn import XGBClassifier
from pydub import AudioSegment
import logging

model=""

def create_timestamps(input_audio_path):
    os.makedirs('./timestamps', exist_ok=True)
    audio_file= input_audio_path
    audio = AudioSegment.from_wav(audio_file)
    list_of_timestamps = [5, 10, 15, 20, 25, 30] #and so on in *seconds*

    start = 0
    for  idx,t in enumerate(list_of_timestamps):
        #break loop if at last element of list
        if idx == len(list_of_timestamps):
            break

        end = t * 1000 #pydub works in millisec
        """ print("split at [ {}:{}] ms".format(start, end)) """
        audio_chunk=audio[start:end]
        audio_chunk.export( "./timestamps/timestamp_{}.wav".format(end), format="wav") 

        start = end  #pydub works in millisec

#csv
def create_csv_header():
    header = ['filename','label','chroma_stft','mel_spectogram','spectral_contrast','tonnetz']
    for i in range(1, 41):
        header.append(f' mfcc{i}')
    return header

def generate_csv_music(csv_header, PATH_TIMESTAMPS, path_csv, file_ext='*.wav'):
    if os.path.exists(path_csv):
        print(f'CSV {path_csv} already exists')
    else:
        sound_cases=[]
        for fn in glob.glob(os.path.join(PATH_TIMESTAMPS, file_ext)): #classical music
            filename = fn.split('\\')[1]
            label = 3
            sound_data=[filename]
            sound_data.append(label)
            sound = f'{PATH_TIMESTAMPS}/{filename}'
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
                
        df = pd.DataFrame(sound_cases, columns = csv_header) 
        df.to_csv(path_csv)
        
def load_data(data):  
    X_TEST = data.drop([data.columns[0],'filename', 'label'],axis=1)#droping first column, filename column and label column 
    Y_TEST = data['label']
    print('X_TEST:', X_TEST.shape)
    print('Y_TEST:', Y_TEST.shape)
    return (X_TEST, Y_TEST)

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return indices

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

def series_prediction_3(output_file_name, sound, label):
    sounds = []
    audio = []
    first_preds = []
    first_percent = []
    second_preds = []
    second_percent = []
    third_preds = []
    third_percent = []
    for i in range(len(sound)): #para imprimir tabela de previsoes
        predictions = model.predict_proba(sound)
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
    fig.write_image(output_file_name + ".png")
    
    print('Mean of timestamps:', mean_first)
    
    return df

def load_models(modelpaths):
    global model
    model = XGBClassifier()
    try:
        for i in modelpaths:
            name = i["name"]
            path = i["path"]
            if (name == "best_model_diana_martins.json"):
                model.load_model(path)
    except: 
        logging.exception("load_models: ")

def run(input_file_path, output_file_name, output_directory_path):
    try:
        create_timestamps(input_file_path)
        csv_header = create_csv_header()
        PATH_CSV_TIMESTAMPS = './features_timestamps.csv' #output csv file name
        PATH_TIMESTAMPS = './timestamps'
        generate_csv_music(csv_header, PATH_TIMESTAMPS, PATH_CSV_TIMESTAMPS)
        DATA_3=pd.read_csv(PATH_CSV_TIMESTAMPS)
        (X_TEST,Y_TEST) = load_data(DATA_3)
        series_prediction_3(X_TEST, Y_TEST, output_file_name)
    except:
        logging.exception("run: ")