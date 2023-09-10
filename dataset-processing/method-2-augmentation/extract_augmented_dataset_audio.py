import os
import csv
import json
import random
from collections import deque
import numpy as np
from pydub import AudioSegment

# There is no random seed needed as we do not split into train / validate / test. All data points contribute towards the test set.

def retrieve_details(filepath, speaker, start_time, num_prev):
    """
    Helper function dealing specifically with GAP dataset by retrieving the previous utterances and end time for a specific utterance.

    :param filepath: Path to the file to be read. For example relative path: './gap-dataset/Transcripts/Transcript Group 1 Feb 8 429.txt'
    :param speaker: Speaker identifier (a colour as per GAP protocol)
    :param start_time: Starting time of the utterance.
    :param num_prev: Number of previous utterances to retrieve.
    :returns: Tuple containing end time of the current utterance and list of previous utterances.
    """
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter='\t')  # assuming the file is tab-separated
        next(reader)  # skip the header
        conversation_history = deque(maxlen=num_prev)
        for row in reader:
            curr_speaker = row[0].split('.')[1]
            if curr_speaker == speaker and row[1] == start_time:
                return row[2], conversation_history
            conversation_history.append(row[3])
        print('Nothing was found for ', filepath, ' at time ', start_time)

def time_to_ms(time_str):
    """
    Convert a time string in the format mm:ss.s to milliseconds.

    :param time_str: Time string to be converted.
    :returns: Converted time in milliseconds.
    """
    minutes, seconds = time_str.split(":")

    # convert minutes and seconds to milliseconds and return the sum
    return int(minutes) * 60000 + float(seconds) * 1000

def get_filepath(number, file_type):
    """
    Given a GAP dataset group number and a file type, retrieve the full path to the corresponding audio or transcript file.

    :param number: Group number as a string.
    :param file_type: Type of the file ('audio' or 'transcript').
    :returns: Full path to the requested file.
    """

    if file_type == 'audio':
        gap_audio_folder_path = '../gap-dataset/Audio'
        for filename in os.listdir(gap_audio_folder_path):
            if not filename.startswith("MP4"):
                continue
            if filename.split(' ')[2] == number:
                return os.path.join(gap_audio_folder_path, filename)
    elif file_type == 'transcript':
        gap_transcript_folder_path = '../gap-dataset/Transcripts'
        for filename in os.listdir(gap_transcript_folder_path):
            if not filename.startswith("Transcript"):
                continue
            if filename.split(' ')[2] == number:
                return os.path.join(gap_transcript_folder_path, filename)
    else:
        print('Invalid file type')




audio_filepath = './Audio/MP4 Group 1 Feb 8 429.mp4.wav' # example

def extract_audio(audio_filepath, start_time, end_time,segment_interval, conversational_history, classification, element=''):
    """
    Firstly this function extracts audio segments from an audio file based on start and end times. 
    Secondly it places the resulting audio file in the interruption-dataset/audio folder.
    Thirdly it adds a dataset entry to the {train / test / validation}_classification_text.txt file which contains the conversational history, classification and link to the audio file.
    Note: the classificaiton text file will be contained within a subdirectory of interruption-dataset corresponding to the seed which was used to split the train / test /validate sets.

    :param audio_filepath: Path to the audio file.
    :param start_time: Starting time for extraction in the format mm:ss.s.
    :param end_time: Ending time for extraction in the format mm:ss.s.
    :param segment_interval: Duration of each segment in milliseconds.
    :param conversational_history: List of previous utterances.
    :param classification: Classification of the audio segment.
    :param element: Optional parameter to specify the dataset element (train, test, or validation) that the audio file should be placed in.
    """

    dataset_dir = os.path.join(os.getcwd(), './aug-interruption-dataset')
    # check if the '/Dataset' directory exists
    if not os.path.exists(dataset_dir):
        # create the '/Dataset' directory
        os.mkdir(dataset_dir)
        os.mkdir(os.path.join(dataset_dir, 'audio'))

    group_number = os.path.basename(audio_filepath).split(' ')[2]  # the group number is always the third element of the file name

    audio = AudioSegment.from_wav(audio_filepath)

    start_time_ms = time_to_ms(start_time)
    end_time_ms = time_to_ms(end_time)
    
    # we determine the number of audio files to save
    duration = end_time_ms - start_time_ms
    num_files = int(duration //segment_interval)

    # we add an iteration to catch the remaining time should segment_interval not divide in fully (most cases)
    remainder = duration % segment_interval
    if remainder > 0:
        num_files += 1

    segment_end_time_ms = start_time_ms 
    for i in range(1, num_files+1):
        # determine the start and end times for this segment
        if i == num_files:
            segment_end_time_ms = end_time_ms
        else:
            segment_end_time_ms = min(segment_end_time_ms +segment_interval, end_time_ms) 
        
        segment = audio[start_time_ms:segment_end_time_ms] # extract segment

        file_name = f"Group {group_number}: {start_time} - {end_time} - {i}.wav"
        output_filepath = os.path.join("./aug-interruption-dataset/audio", file_name)

        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        segment.export(output_filepath, format="wav")
        
    
        dataset_path = f'./aug-interruption-dataset'
        os.makedirs(dataset_path, exist_ok=True)
        with open(os.path.join(dataset_path, f'{element}_classification_details.txt'), 'a') as f:
            f.write(f"{file_name}||{classification}||{conversational_history}\n")

def create_dataset(segment_length, num_prev):
    """
    Creates a dataset by splitting the data into train, test, and validation sets, and extracting audio segments using the extract_audio helper function.

    :param segment_length: Length of each audio segment in milliseconds.
    :param num_prev: Number of previous utterances to retrieve.
    """

    # loop through all instances of aug_data.json
    with open('aug_data.json', 'r') as f:
        train_data = json.load(f) # we only add to the training set, no need for splitting into validation and test sets
    

    # split into dictionary so we can process each element separately
    datasets = { 'train': train_data }

    for element, dataset in datasets.items():
        for annotation in dataset:
            start_time = annotation['startTime']
            classification = annotation['classification']
            audio_filepath = get_filepath(annotation['groupNumber'], 'audio')
            transcript_filepath = get_filepath(annotation['groupNumber'], 'transcript')
            end_time, conversational_history = retrieve_details(transcript_filepath, annotation['speakerId'], start_time, num_prev)
            extract_audio(audio_filepath, start_time, end_time, segment_length, conversational_history[0], classification, element=element)



if __name__ == "__main__":
    create_dataset(300, 1)