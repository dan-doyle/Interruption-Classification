This repository is split into two sections:

# 1. Dataset

## Directory structure

The Data Processing subdirectory contains the files required for generating the Interruption Dataset from the GAP Dataset. The parse_transcript.py script retrieves instances of overlapped speech, allowing us to manually classify and store them in '/Manual Annotations/data.json'. The extract_dataset_audio.py script then produces audio snippets and their corresponding text files with classification details.

The method-2-augmentation subdirectory contains similar scripts for the extraction of further audio data from backchannels in the GAP Dataset.

In the folder structure diagram below, we indicate which dataset folders are left empty for space purposes. Given the GAP Dataset as a starting point, all of these folders can be created from the scripts contained in this repository.

```
|- Dataset/
|  |- Data Processing/
|  |  |- Manual Annotations/
|  |  |  --> data.json
|  |  |- Overlap Transcripts/ <- Left empty
|  |  --> extract_dataset_audio.py
|  |  --> parse_transcript.py
|  |- GAP Dataset/
|  |  |- Audio/ <- Left empty
|  |  |- Transcripts/ <- Left empty
|  |- Interruption Dataset/ <- Left empty
|  |- method-2-augmentation/
|  |  --> aug_data.json
|  |  --> extract_augmented_dataset_audio.py
|  |  --> process_overlap_transcript.py
|  --> generate_embeddings.ipynb
|  --> print_data_stats.py
```

## Set-up
In creating our dataset we navigate to the Data Processing folder and run parse_transcript.py followed by extract_dataset_audio.py. The latter requires the [pydub](https://github.com/jiaaro/pydub) which relies on [Ffmpeg](https://ffmpeg.org/). Please first install Ffmpeg and then run the command:

```
pip install pydub
```

Following this, we can use the generate_embeddings.ipynb to create embeddings from the audio snippets in the dataset.

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

Each of the IPython notebook is ready to be run in a Jupyter notebook environment so long as we install the directories required (as listed in the first code block). Additionally they can be run in a Google Colab environment by simply uncommenting the 'pip installs' in the first code block.