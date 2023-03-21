# Bangla-Complex-Named-Entity-Recognition-Challenge

Winning Solution for the **Bangla Complex Named Entity Recognition Challenge** - BDOSN NLP Hackathon 2023 [[arxiv](https://arxiv.org/abs/2303.09306)]

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

*Read the [[arxiv report](https://arxiv.org/abs/2303.09306)] for more details.*

# **Results**

## `Feature Based Track`

|                              Feature                                  |    F1 Score   |
|:----------------------------------------------------------------:     |:-------------:|
|                        POS Tagger, Suffix                             |     0.56      |
|               POS Tagger, Suffix, k-Neighbor Words                    |     0.62      |
|       POS Tagger, Suffix, k-Neighbor Words, Gazetteer Lists           |     0.689     |
|           POS Tagger, Prefix, Suffix, k-Neighbor Words                |     0.692     |
| **POS Tagger, Prefix, Suffix, k-Neighbor Words, k-means clustering**  |   **0.72**    |

## `DL Based Track`

| **Model**                     | **Batch Size** | **Max Seq Length** | **Epoch** | **F1 Score** |
|:-----------------------------:|:--------------:|:------------------:|:---------:|:-------------:|
| base                          | 16             | 128                | 3         | 0.73          |
| large                         | 16             | 128                | 3         | 0.77          |
| large                         | 32             | 64                 | 3         | 0.76          |
| large                         | 16             | 128                | 6         | 0.78          |
| large                         | 32             | 64                 | 6         | 0.79          |
| oversampled+large             | 16             | 128                | 6         | 0.78          |
| SemEval2023data+large         | 32             | 64                 | 4         | 0.78          |
| SemEval2023data+weights+large | 32             | 64                 | 4         | 0.74          |
| **SemEval2023data+large**         | 32             | 64                 | 6         | **0.79**          |


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

# **Citation**
[[arxiv](https://arxiv.org/abs/2303.09306)]

```bibtex
@misc{shahgir2023banglaconer,
      title={BanglaCoNER: Towards Robust Bangla Complex Named Entity Recognition}, 
      author={HAZ Sameen Shahgir and Ramisa Alam and Md. Zarif Ul Alam},
      year={2023},
      eprint={2303.09306},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
