This repository is split into two sections:

# 1. Data Processing

## Directory structure

The data processing directory contains the files required for generating the Interruption Dataset from the GAP Dataset (which is located in the gap-dataset subdirectory). The parse_transcript.py script retrieves instances of overlapped speech, allowing us to manually classifying and store them in data.json. The extract_dataset_audio.py then produces audio snippets and their corresponding text files with classification details.

The method-2-augmentation subdirectory contains similar scripts for the extraction of further audio data from backchannels in the GAP Dataset.

```
|- dataset-processing/
|  --> data.json
|  --> extract_dataset_audio.py
|  |- gap-dataset/
|  |  |- Audio/
|  |  |- Transcripts/
|  --> generate_embeddings.ipynb
|  |- method-2-augmentation/
|  |  --> aug_data.json
|  |  --> extract_augmented_dataset_audio.py
|  |  --> process_overlap_transcript.py
|  --> parse_transcript.py
|  --> print_data_stats.py
```

## Set-up
In creating our dataset we run parse_transcript.py followed by extract_dataset_audio.py. The latter requires the [pydub](https://github.com/jiaaro/pydub) which relies on [Ffmpeg](https://ffmpeg.org/). Please first install Ffmpeg and then run command:

```
pip install pydub
```

# 2. Modelling

## Directory structure

Our modelling is broken down into three sections which each have a corresponding folder: 
- The baseline folder contains the VAD baseline approach.
- The average_based folder contains the Average-based audio and multi-modal approaches. This also includes a grid-search analysis notebook for the audio model.
- The pattern_based folder contains the LSTM and TCN models where the latter has a corresponding grid-search notebook.

```
|- modelling/
|  |- average_based/
|  |  --> grid_search_average_model.ipynb
|  |  --> test_average_audio_model.ipynb
|  |  --> test_average_multimodal_model.ipynb
|  |  --> train_average_audio_model.ipynb
|  |  --> train_average_multimodal_model.ipynb
|  |- baseline/
|  |  --> test_VAD.ipynb
|  |  --> train_VAD.ipynb
|  |- pattern_based/
|  |  --> grid_search_TCN_model.ipynb
|  |  --> test_LSTM_model.ipynb
|  |  --> test_TCN_model.ipynb
|  |  --> train_LSTM_model.ipynb
|  |  --> train_TCN_model.ipynb
```

## Set-up

Each of the IPython notebooks are ready to be run in a Jupyter notebook environment so long as we install the directories required (as listed in the first code block). Additionally they can be run in a Google Colab environment by simply uncommenting the 'pip installs' in the first code block.