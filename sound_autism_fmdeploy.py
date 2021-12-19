#imports
import os
import pandas as pd
import glob
import numpy as np
import librosa
import plotly.graph_objs as go
from xgboost.sklearn import XGBClassifier
from pydub import AudioSegment
from pydub.utils import make_chunks
import logging

def create_timestamps(input_audio_path, output_directory_path):
    try:
        os.makedirs(output_directory_path + 'timestamps', exist_ok=True)
        audio_file= input_audio_path
        audio = AudioSegment.from_wav(audio_file)
        chunk_length_ms = 5000 # pydub calculates in millisec
        chunks = make_chunks(audio, chunk_length_ms) # make chunks of 5s
        # export all of the individual chunks as wav files
        for i,chunk in enumerate(chunks):
            chunk_name = output_directory_path + "timestamps/chunk{0}.wav".format(i)
            chunk.export(chunk_name, format="wav")
    except: 
        logging.exception("create_timestamps: ")

#csv
def create_csv_header():
    try:
        header = ['filename','label','chroma_stft','mel_spectogram','spectral_contrast','tonnetz']
        for i in range(1, 41):
            header.append(f' mfcc{i}')
        return header
    except: 
        logging.exception("create_csv_header: ")

def generate_csv_music(csv_header, PATH_TIMESTAMPS, path_csv, file_ext='*.wav'):
    try:
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
    except: 
        logging.exception("generate_csv_music: ")
        
def load_data(data):  
    try:
        X_TEST = data.drop([data.columns[0],'filename', 'label'],axis=1)#droping first column, filename column and label column 
        Y_TEST = data['label']
        print('X_TEST:', X_TEST.shape)
        print('Y_TEST:', Y_TEST.shape)
        return (X_TEST, Y_TEST)
    except: 
        logging.exception("load_data: ")

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    try:
        flat = ary.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return indices
    except: 
            logging.exception("largest_indices: ")

def truncate(n, decimals=0):
    try:
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier
    except: 
        logging.exception("truncate: ")

def series_prediction(model, output_file_name, output_directory_path, sound, label):
    try:
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
            header=dict(values=list(df.columns)),
            cells=dict(values=[df.Y_test, df.first_pred, df.first_percent, df.second_pred, # df.Sound,df.mean_first,df.mean_second,df.mean_third
                            df.second_percent, df.third_pred, df.third_percent],
                    )
        )])
        """ fig.show() """
        fig.write_image(output_directory_path + output_file_name + ".png")
        df.to_csv(output_directory_path + output_file_name + ".csv")
        """ print('Mean of timestamps:', mean_first) """
    except: 
        logging.exception("series_prediction: ")
        
def input_validation(input_file_path):
    try:
        if (input_file_path.lower().endswith(('.wav'))):
            logging.debug('file is .wav')
        else:
            raise ValueError('file extension error: file must be .wav')
    except Exception as e:
        logging.error(str(e))
        
def load_models(modelpaths):
    global model
    model = XGBClassifier()
    try:
        for i in modelpaths:
            name = i["name"]
            path = i["path"]
            if (name == "sound_autism_model.json"):
                model.load_model(path)
    except: 
        return logging.exception("load_models: ")

def run(input_file_path, output_file_name, output_directory_path):
    try:
        input_validation(input_file_path)
        create_timestamps(input_file_path, output_directory_path)
        csv_header = create_csv_header()
        PATH_CSV_TIMESTAMPS = output_directory_path + 'features_timestamps.csv' #output csv file name
        PATH_TIMESTAMPS = output_directory_path + 'timestamps'
        generate_csv_music(csv_header, PATH_TIMESTAMPS, PATH_CSV_TIMESTAMPS)
        DATA_3=pd.read_csv(PATH_CSV_TIMESTAMPS)
        (X_TEST,Y_TEST) = load_data(DATA_3)
        series_prediction(model, output_file_name, output_directory_path, X_TEST, Y_TEST)
    except:
        return logging.exception("run: ")