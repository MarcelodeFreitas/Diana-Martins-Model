from pydub import AudioSegment
import logging

def input_validation(input_file_path):
    try:
        if (input_file_path.lower().endswith(('.mp3'))):
            logging.debug('file is .mp3')
        else:
            raise ValueError('file extension error: file must be .mp3')
    except Exception as e:
        logging.exception("input_validation: " + str(e))

def convert_mp3_to_wav(input_file_path, output_file_name, output_directory_path):
    try:
        sound = AudioSegment.from_mp3(input_file_path)
        sound.export(output_directory_path + output_file_name + ".wav", format="wav")
    except Exception as e:
        logging.exception("convert_mp3_to_wav: " + str(e))

def run(input_file_path, output_file_name, output_directory_path):
    try:
        input_validation(input_file_path)
        convert_mp3_to_wav(input_file_path, output_file_name, output_directory_path) 
    except:
        return logging.exception("run: ")