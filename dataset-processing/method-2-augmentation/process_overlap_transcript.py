import os
import re
import time
import openai

# ensure we set the API key as an environment variable before running the script
TRANSCRIPT = '' # set to transcript we wish to process e.g. Group 28: current_overlaps.txt
openai_key = os.environ.get('OPENAI_KEY')
if not API_KEY:
    raise ValueError("API key not found in environment variables.")

augment_filepath = './aug_overlap_transcripts'
overlap_transcript_file = 'Group 28: current_overlaps.txt'
file_path = os.path.join(augment_filepath, overlap_transcript_file)

def filter_snippets(content):
    """
    Filters the provided content to remove snippets that have '[laughter]' or '[noise]'
    in their last two lines.

    :param content: List of snippets where each snippet is a string of text.
    :returns: List of filtered snippets where the final utterance in each snippet has " <- evaluate here" appended to focus attention on this line.
    """
    result = []
    for s in content:
        lines = s.strip().split("\n")
        if len(lines) >= 2: 
            last_two_lines = lines[-2:]
            if not any(substring in line for line in last_two_lines for substring in ['[laughter]', '[noise]']):
                result.append(s + " <- evaluate here")
    return result

def extract_snippets(filename):
    """
    Given a file produced by parse_transcript as an input, we extract overlapped speech ('snippets'), process them using the filter_snippets helper function and group into groups of three.

    :param filename: Name of the file to read snippets from.
    :returns: List of grouped snippets. Each group contains up to three snippets.
    """
    with open(filename, 'r') as file:
        content = file.read().strip().split("\n\n")
        # filter out instance of overlapped speech if second last turn or last turn contains laughter or noise
    content = filter_snippets(content)
    # we group the instances of overlapped speech into threes
    grouped_snippets = []
    for i in range(0, len(content), 3):
        group = content[i:i+3]
        grouped_string = '\n\n'.join(group)
        grouped_snippets.append(grouped_string)

    return grouped_snippets

def extract_snippet(snippets, timestamp):
    """
    Finds and returns the snippet of overlapped speech which contains the given timestamp.

    :param snippets: String containing multiple snippets separated by double newline.
    :param timestamp: Timestamp string to search for.
    :returns: Single snippet string containing the timestamp or None if not found.
    """
    snippets = snippets.split("\n\n")
    for snippet in snippets:
            if timestamp in snippet:
                return snippet
    return None

def process_final_answer(final_reply):
    """
    Processes the second and final LLM reply and extracts boolean answer from the response.

    :param final_reply: String containing the reply.
    :returns: Boolean indicating whether the enclosed answer is true or not.
    """
    pattern = r'<answer>(.*?)<\/answer>'
    match = re.search(pattern, final_reply, re.DOTALL)
    if match:
        content = match.group(1).strip()
        if content == "true" or content == "True":
            return True
    return False

def extract_answer_content(s):
    """
    Extracts the timestamp inside <answer> tags from the given string if a backchannel is present, otherwise None is returned.

    :param s: String representing the LLM's second and final reply.
    :returns: Content inside <answer> tags if found and matches the timestamp format, otherwise None.
    """
    pattern = r'<answer>(.*?)<\/answer>'
    match = re.search(pattern, s, re.DOTALL)
    # check to see if appropriate format, if not we return None
    if match:
        content = match.group(1).strip()  # Stripping in case there are leading/trailing whitespaces
        if re.match(r'^\d{2}:\d{2}\.\d$', content):
            return content
    return None

def add_sample_to_list(data, timestamp):
    """
    Appends a new data sample to the provided list.

    :param data: List to which the sample will be added.
    :param timestamp: Timestamp string to be set as the 'startTime' of the new data sample.
    :returns: None
    """
    new_data = {
        'startTime': timestamp,
        'classification': 'non-interruption'
    }
    data.append(new_data)

def add_data_to_json(data, filename):
    """
    Appends provided data to the JSON augmented dataset file.

    :param data: List of data samples to append.
    :param filename: Name of the JSON augmented dataset file.
    :returns: None
    """
    with open('aug_data.json', 'r') as json_file:
        aug_data_list = json.load(json_file)
    aug_data_list.extend(data)
    with open('aug_data.json', 'w') as json_file:
        json.dump(aug_data_list, json_file)

first_prompt = """You are given three conversational snippets below separated by spaces and each starting with a timestamp. In each snippet an overlap of speech occurs between the last and second last turn, the last turn (where the <- evaluate this is) is where we are evaluating a potential backchannel. Please return only the timestamp of the backchannel, if any are, enclosed in <answer></answer> tags and give your reasoning beforehand in <reflection></reflection> tags. Only return an answer if confident.

Backchannel is defined as as a brief affirmation by a listener to what a speaker is saying through words or noises (for example, "agreed", "sure" or "mhmm"). Backchannel has no intent to take the turn of the conversation. Be careful of cases that sound like backchannel but are actually an answer to a question, for example where a 'yeah' overlaps in response to 'would you like a hotdog'. This is an answer to this question and takes the turn of the conversation. From this we can see that context is important in determining whether it is backchannel or not. 

Here are the snippets:
{}
"""

second_prompt = """In the below snippet an overlap of speech occurs between the last and second last turn, the last turn (where the <- evaluate this is) is where we are evaluating a potential backchannel. Please return true or false enclosed in <answer></answer> tags and give your reasoning beforehand in <reflection></reflection> tags. Only return an answer if confident.

Backchannel is defined as as a brief affirmation by a listener to what a speaker is saying through words or noises (for example, "agreed", "sure" or "mhmm"). Backchannel has no intent to take the turn of the conversation. Be careful of cases that sound like backchannel but are actually an answer to a question, for example where a 'yeah' overlaps in response to 'would you like a hotdog'. This is an answer to this question and takes the turn of the conversation. From this we can see that context is important in determining whether it is backchannel or not. 

Here is the snippet:
{}
"""

def query_gpt(prompt, model_="gpt-4", max_retries=10, initial_delay=1):
    delay = initial_delay

    # implementing exponential back-off
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model_,
                messages=[
                    {"role": "system", "content": "You are an English teacher."},
                    {"role": "user", "content": prompt}
                ],
                api_key=openai_key
            )

            # extract the generated reply
            reply = response.choices[0].message.content

            return reply

        except (openai.error.ServiceUnavailableError, openai.error.RateLimitError) as e:
            if attempt < max_retries - 1:  # no delay needed after the last attempt
                print(f"Server is overloaded, waiting for {delay} seconds before retrying.")
                time.sleep(delay)
                delay *= 2  # double the delay
            else:
                raise # re-raise the last exception if all attempts failed

if __name__ == "__main__":
    snippets = extract_snippets(file_path)
    data = []
    # loop through snippets of overlapped speech calling query_gpt
    for snippet in snippets:
        potential_answer = query_gpt(first_prompt.format(snippet))
        timestamp = extract_answer_content(potential_answer)
        # logic: if answer not None: we extract the relevant snippet and query one
        if timestamp:
            potential_noninterruption = extract_snippet(snippet, timestamp)
            if potential_noninterruption:
                print('This is inserted into the second prompt: ', potential_noninterruption)
                final_reply = query_gpt(second_prompt.format(potential_noninterruption))
                final_answer = process_final_answer(final_reply)
                if final_answer:
                    add_sample_to_list(data, timestamp)
                    print('Added the following timestamp to the report: ', timestamp)
    add_data_to_json(data, './Automatic Annotations/' + overlap_transcript_file)