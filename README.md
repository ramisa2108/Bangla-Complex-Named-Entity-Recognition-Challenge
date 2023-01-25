# Bangla-Complex-Named-Entity-Recognition-Challenge

Winning Solution for the **Bangla Complex Named Entity Recognition Challenge** - BDOSN NLP Hackathon 2023

# **Dataset**

The provided dataset was a labeled Bangla NER dataset in the `ċonll` format where each word had a corresponding NER tag and sentences were separated with empty lines. The `train` set consists of *15300* sentences and the `validation` set has *800* sentences. The length of the sentences in both sets varies from *2 words to 35 words* with the average length being 12 words. There are 7 different NER tags in the given dataset.

| Tags | Count |
|:----:|:-----:|
|  `LOC` |  3804 |
|  `GRP` |  6653 |
| `PROD` |  5152 |
|  `CW`  |  5001 |
| `CORP` |  5299 |
|  `PER` |  6738 |
|   `O`  |  170K |

Presence of `CW`, `PROD`, `CORP` and `GRP` tags in the dataset makes the task challenging.

The dataset is available in the [`data`](/data/) folder.

# **Approach**

The competition had two tracks, one was a DL based track and the other was a feature based track. We participated in both the tracks and our solution for the DL based track was based on the [Bangla BERT](https://github.com/csebuetnlp/banglabert) architecture and our solution for the feature based track was based on the [CRF](https://en.wikipedia.org/wiki/Conditional_random_field) architecture.

# **Reproducing the Results**

## Running DL Model

### Normalizer (Required)
``` 
$ pip install git+https://github.com/csebuetnlp/normalizer
```
### New Data
https://multiconer.github.io/competition

2023 Train and Dev Datasets (about 10K)


**train_inference.py:**

The train_file_path, validation_file_path need to be set inside the main function and the varialble 'train' need to be set to True to train.

## Running Feature Based Model

**bangla-crf-baseline.ipynb and bangla-crf-with-kmeans-and-gazetteer.ipynb:**


The files included in the data folder should remain in a relative path "../data" for running the notebooks.
