from collections import deque
from datetime import datetime, timedelta
import csv


class ConversationIterator:
    """
        Receives a Group Affect and Performance Transcript file and iterates through the conversation identifying cases of overlapped speech relevant to our research. This class implements the iterator protocol and with every iteration this class returns a case of overlapped speech.

        Edge cases:
            - Process all available utterances when context-length isn't available at the beginning
            - First utterance cannot be an interruption hence is not returned
    """
    def __init__(self, data_filepath, context_length = 5):
        """
        :param data_filepath: Path to the GAP file
        :param context_length: Number of context utterances to provide when returning a case of overlapped speech
        """
        self.context_length = context_length
        self.data = self.parse_GAP_file(data_filepath)
        self.conversation_history = deque(maxlen=self.context_length)
        self.current_speaker = None
        self.current_timestamp = None
        self.prev_timestamp = None
        self.current_overlap = False
        self.short_pause = False
        self.index = 0
    
    def __iter__(self):
        """
        Returns the iterator instance.
        
        :returns: Iterator instance
        """
        return self
    
    def __next__(self):
        """
        Returns the next instance of overlapped speech, or if none remaining the iterator is stopped. 
        
        :returns: The next conversational turn
        :raises StopIteration: If there are no more conversational turns
        """
        conversation = self.get_conversation()
        if conversation is None:
            raise StopIteration
        return conversation

    def parse_GAP_file(self, filename):
        """
        Preprocesses the GAP files to extract utterances into an ordered array format

        :param filename: Path to the GAP file
        :returns: Parsed data from the GAP file
        """
        data = []
        with open(filename, 'r') as f:
            # transcript files are tab-separated
            reader = csv.reader(f, delimiter='\t') 
            # skip the header in the Transcript file
            next(reader) 
            users = {}
            for row in reader:
                data.append(row)
                speaker_colour = row[0].split('.')[1]
                if speaker_colour not in users:
                    users[speaker_colour] = len(users) + 1
                data[-1][0] = users[speaker_colour]
        return data

    def _process_utterance(self, utterance, include_speaker):
        """
        Formats each utterance adding in textual details about the speaker, noises and pauses

        :param utterance: The utterance to be processed
        :param include_speaker: Flag indicating if the speaker identifier should be included in the result, in cases where two consecutive utterances result from the same speaker we only need one 'Speaker' included
        :returns: Processed utterance
        """
        participant = utterance[0]
        utterance[3] = utterance[3].replace('$', '[laughter]')
        overlap = ''
        short_pause = '[short pause] ' if self.short_pause else ''
        return "\nSpeaker " + f'{participant}: {overlap}{short_pause}{utterance[3]}' if include_speaker else f' {utterance[3]}'

    def get_conversation(self):
        """
        Retrieves the conversation with the next utterance from the conversation included and indicates if the last utterance has overlapped speech, using the 'check_overlap' helper method. 
        This method changes the state of the class to ensure we move through the transcript in a chronological fashion

        :returns: Current conversational turn or None if there are no more turns
        """
        if self.index < len(self.data): 
            unprocessed_utterance = self.data[self.index]
            self.current_speaker = unprocessed_utterance[0]
            self.current_timestamp = unprocessed_utterance[1:3]
            if self.index > 0:
                self.current_overlap = self.check_overlap()
                self.short_pause = self.check_pause(unprocessed_utterance[1])
            utterance = self._process_utterance(unprocessed_utterance, True)
            self.conversation_history.append(utterance)
            self.index += 1

            result = {}
            # remove speech that is not attributed to a speaker
            while 'Speaker' not in self.conversation_history[0]:
                self.conversation_history.popleft()
            result['transcript'] = ''.join(list(self.conversation_history))
            result['timestamp'] = self.current_timestamp
            result['overlap'] = self.current_overlap
            self.current_overlap = False
            self.short_pause = False
            self.prev_timestamp = self.current_timestamp
            return result
        else:
            return None     

    def check_overlap(self):
        """
        Checks if the current utterance overlaps with the previous.

        :returns: True if overlaps, otherwise False
        """
        # extract this processing code to a helper as it is used multiple times
        prev_end_time = self.prev_timestamp[1]
        prev_start_time = self.prev_timestamp[0]
        curr_end_time = self.current_timestamp[1]
        curr_start_time = self.current_timestamp[0]

        # convert time strings to datetime objects
        prev_start_datetime = datetime.strptime(prev_start_time, "%M:%S.%f")
        prev_end_datetime = datetime.strptime(prev_end_time, "%M:%S.%f")
        curr_start_datetime = datetime.strptime(curr_start_time, "%M:%S.%f")
        curr_end_datetime = datetime.strptime(curr_end_time, "%M:%S.%f")

        # we only count overlaps that are not in the first 300ms or last 10% of speech
        basic_overlap_condition = prev_start_datetime < curr_end_datetime
        immediate_overlap_condition = (curr_start_datetime - prev_start_datetime).total_seconds() > 0.3

        prev_utterance_duration = prev_end_datetime - prev_start_datetime
        early_onset_deadline = prev_end_datetime - (prev_utterance_duration / 10)
        not_early_onset_condition = curr_start_datetime < early_onset_deadline

        if basic_overlap_condition and immediate_overlap_condition and not_early_onset_condition:
            return True
        return False

    
    def check_pause(self, new_time):
        """
        Checks if there was a pause before the next utterance. A pause is deemed to have occurred if there is a gap of greater than two seconds between consecutive utterances.

        :param new_time: Time of the next utterance
        :returns: True if there was a pause, otherwise False
        """
        curr_time = self.current_timestamp[1]

        curr_datetime = datetime.strptime(curr_time, "%M:%S.%f")
        new_datetime = datetime.strptime(new_time, "%M:%S.%f")
        time_difference = new_datetime - curr_datetime

        if time_difference > timedelta(seconds=2):
            return True
        return False


if __name__ == "__main__":
    output = ""
    transcript_filepath = "./gap-dataset/Transcripts/Transcript Group 1 Feb 8 429.txt"

    c = ConversationIterator(transcript_filepath)    
    i = 1
    overlap_count = 0
    for curr in c:
        if curr['overlap']:
            output += curr['timestamp'][0]
            output += curr['transcript'] + '\n\n'
            overlap_count += 1
        i += 1

    print('OVERLAP COUNT:', overlap_count, 'Out of:', i)

    with open('current_overlaps.txt', 'w') as file:
        file.write(output)