# DA-6401_Assignment_3
# Overview
The purpose of this assignment was:
1. Building and training a RNN model from scratch for seq2seq character level Neural machine transliteration.
2. Implement attention based model.

The link to the wandb project runs:
https://wandb.ai/ma23c011-indian-institute-of-technology-madras/transliteration-sweep/table?nw=nwuserma23c011

The link to the wandb report:
https://wandb.ai/ma23c011-indian-institute-of-technology-madras/transliteration-sweep/reports/DA6401-Assignment-3--VmlldzoxMjgyODMzOQ
## Dataset:

The dakshina dataset released by google was used for 
In this assignment the Dakshina dataset(https://github.com/google-research-datasets/dakshina) released by Google has been used. This dataset contains pairs of the following form: 
﻿xxx.      yyy﻿
ajanabee अजनबी.
i.e., a word in the native script and its corresponding transliteration in the Latin script (the way we type while chatting with our friends on WhatsApp etc). Given many such (xi,yi)i=1n(x_i, y_i)_{i=1}^n(xi​,yi​)i=1n​ pairs your goal is to train a model y=f^(x)y = \hat{f}(x)y=f^​(x) which takes as input a romanized string (ghar) and produces the corresponding word in Devanagari (घर). 
These blogs were used as references to understand how to build neural sequence to sequence models: 

https://keras.io/examples/nlp/lstm_seq2seq/
https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/

By default the implemented model uses telugu as the target language. 

## Building and training a RNN model with and without attention from scratch for sequence to sequence character level neural machine transliteration:

A model class has been implemented in ```model.py``` which provides a variety of options to build RNN models and add dropout,hidden layers, neurons, etc. 
The training, and testing scripts can be found in ```train.py```
System requirements are as follows:
- ```CUDA == 11.0```
- ```CUDNN >= v8.0```
- ```Python >= 3.6 ```


A Bahdanau based attention has been implemented by adapting the code for the bahdanau attention layer class from : https://github.com/thushv89/attention_keras. 
It is advised to setup a virtual environment if running locally using virtualenv/venv and pyenv for python version handling. Or even better, use Conda. But in this assignent I have not used anaconda package manager. 
### Training:
Wandb framework is used to track the loss and accuracy metrics of training and validation. Moreover, bayesian sweeps have been performed for various hyper parameter configurations. 
The sweep configuration and default configurations of hyperparameters are specficied as follows:
```
sweep_config = {
    "method"﻿: "grid"﻿,
    "parameters"﻿: {
        "enc_dec_layers": {
           "values": [1, 2, 3]
        },
        "units": {
            "values": [64, 128, 256]
        },
        "layer_type": {
            "values": ["rnn", "gru", "lstm"]
        }
        "enc_dec_layers": {
           "values": [2, 3]
        },
        "embedding_dim": {
            "values": [64, 128, 256]
        },
        "dropout": {
            "values": [0.2, 0.3]
        }
        "beam_width": {
            "values": [3, 5, 7]
        }
        "teacher_forcing_ratio": {
            "values": [0.3, 0.5, 0.7, 0.9]
        },
        "enc_dec_layers": {
            "values": [2]
        },
        "embedding_dim": {
            "values": [128]
        },
        "dropout": {
            "values": [0.2]
        }
        "attention": {
            "values": [True, False]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project='transliteration-sweep')

# The following is placed within the train() function. 
config_defaults = {"embedding_dim": 64,
                    "enc_dec_layers": 1,
                    "layer_type": "lstm",
                    "units": 128,
                    "dropout": 0,
                    "attention": False,
                    "beam_width": 3,
                    "teacher_forcing_ratio": 1.0
                    }
 
```

The user can either run the colab notebooks directly or use the python script:
The commands to run the training script is simply:

```python3 trian.py```


### Testing:

In order to test the best trained model on the test data set, a test script has been written that:
1. Evaluates the test accuracy
2. Saves the predicitons in a csv file




### Hyperparameter sweeps:

One can find colab notebooks which are self contained and they can be run on a GPU based runtime session and the results will be logged accordingly in the user entity's wandb account which alone needs to be changed in the notebook before beginning the run. 

### With attention code can be used by putting True in Hyperparameter


