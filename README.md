# Memory-Augmenting Decoder-Only Language Models through Encoders
This repository is relative to the paper Memory-Augmenting Decoder-Only Language Models through Encoders. DOI coming soon...

## Folder structure
```bash
.
├── data                              # Includes one folder for each dataset used with scripts to download and pre-process them
├── llama_memory
│   ├── llama                         # Code of the LLaMA
│   │   ├── model.py                  # Changes were made only to this file
│   │   └── ...
│   ├── test.py                       # A script to test the performance of the model and some simple baselines
│   ├── trainer.py                    # A script for training the model
│   └── ...
├── build_all_datasets.sh             # Script to download and pre-process all the datasets
└── ...
```

## Datasets
Following are the datasets used and their source. In each folder under [data/](data/) there is a script to download and pre-process each dataset.

| Name | Memory content | Goal |
| --- | --- | --- |
| Curiosity (curio) | (Fun?) facts | Conversations on topic |
| DREAM (dream) | A dialogue | Question about dialogue |
| SGD (sgd) | Different info to base the answer on | Costumer service or similar |
| Wizard of Wikipedia (wow) | Snippets of Wiki | Conversations |

## Acknowledgements
Part of this code was borrowed from [facebookresearch/llama](https://github.com/facebookresearch/llama/tree/main). Part of this code was inspired by/adapted from [OpenGVLab/LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter).

## License
As this work is based on the first version of LLaMA, it inherits its GPLv3 license.
