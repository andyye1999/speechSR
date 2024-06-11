# 语音超分辨率

## 整体结构
共有两个模型 时域方法和频域方法

文件总共有几种  
config配置文件：配置训练以及验证的模型 dataset等  
dataset  训练和测试的dataset  
model 和 modules  包含模型  
trainer文件夹 训练框架  
soundstream文件夹 本文参考了aicodec 因此用到soundstream模型  
util 包含一些可视化代码  
train.py  训练主程序  
txt.py  用于生成训练音频配置文件txt   
txttest.py 用于生成测试音频配置文件txt   
LSDSISDR.py  用于简单的测试指标  
`enhancementdpranbwe48.py` or `enhancementseanetbwe48.py`  测试用

## pip list

```yaml


Package                        Version  
------------------------------ --------------------
absl-py                        2.0.0               
audioread                      3.0.1               
auraloss                       0.4.0               
bottleneck-transformer-pytorch 0.1.4               
Brotli                         1.0.9               
cachetools                     5.3.2               
certifi                        2023.11.17          
cffi                           1.16.0              
charset-normalizer             3.3.2               
colorama                       0.4.6               
contourpy                      1.1.1  
cycler                         0.12.1  
decorator                      5.1.1  
dominate                       2.9.0  
einops                         0.7.0  
filelock                       3.13.1  
fonttools                      4.45.1  
fsspec                         2023.12.1  
Gammatone                      1.0  
google-auth                    2.23.4  
google-auth-oauthlib           1.0.0  
grpcio                         1.59.3  
huggingface-hub                0.19.4  
idna                           3.6  
importlib-metadata             6.8.0  
importlib-resources            6.1.1  
joblib                         1.3.2  
json5                          0.9.14  
kiwisolver                     1.4.5  
lazy_loader                    0.3  
librosa                        0.10.1
llvmlite                       0.41.1
local-attention                1.9.0
Markdown                       3.5.1
MarkupSafe                     2.1.3
matplotlib                     3.7.4
mock                           5.1.0
msgpack                        1.0.7
nose                           1.3.7
numba                          0.58.1
numpy                          1.24.3
oauthlib                       3.2.2
packaging                      23.2
pesq                           0.0.4
Pillow                         9.2.0
pip                            23.3.1
platformdirs                   4.0.0
pooch                          1.8.0
protobuf                       4.25.1
pyasn1                         0.5.1
pyasn1-modules                 0.3.0
pycparser                      2.21
pyparsing                      3.1.1
pysepm                         0.1
PySocks                        1.7.1
pystoi                         0.3.3
python-dateutil                2.8.2
PyYAML                         6.0.1
requests                       2.31.0
requests-oauthlib              1.3.1
rsa                            4.9
safetensors                    0.4.1
scikit-learn                   1.3.2
scipy                          1.10.1
setuptools                     68.0.0
six                            1.16.0
soundfile                      0.12.1
sox                            1.4.1
soxr                           0.3.7
SRMRpy                         1.0
tensorboard                    2.14.0
tensorboard-data-server        0.7.2
thop                           0.1.1.post2209072238
threadpoolctl                  3.2.0
timm                           0.9.12
torch                          1.12.1
torchaudio                     0.12.1
torchlibrosa                   0.1.0
torchvision                    0.13.1
tqdm                           4.66.1
typing_extensions              4.8.0
urllib3                        2.1.0
Werkzeug                       3.0.1
wheel                          0.41.2
win-inet-pton                  1.1.0
zipp                           3.17.0

```
### Training

Use `train.py` to train the model. It receives three command line parameters:

- `-h`, display help information
- `-C, --config`, specify the configuration file required for training
- `-R, --resume`, continue training from the checkpoint of the last saved model

Syntax: `python train.py [-h] -C CONFIG [-R]`

E.g.:

```shell script
python train.py -C config/cbbrn1bwe16_48/cbbrn1bwe16_48.json
# The configuration file used to train the model is "config/cbbrn1bwe16_48/cbbrn1bwe16_48.json"
# Use all GPUs for training

python train.py -C config/dpranbwe16to48/dpranbwe16_48.json
```

### Enhancement

Use `enhancementdpranbwe48.py` or `enhancementseanetbwe48.py` to enhance noisy speech, which receives the following parameters:

- `-h, --help`, display help information
- `-C, --config`, specify the model, the enhanced dataset, and custom args used to enhance the speech.
- `-D, --device`, enhance the GPU index used, -1 means use CPU
- `-O, --output_dir`, specify where to store the enhanced speech, you need to ensure that this directory exists in advance
- `-M, --model_checkpoint_path`, the path of the model checkpoint, the extension of the checkpoint file is .tar or .pth

Syntax: `python enhancement.py [-h] -C CONFIG [-D DEVICE] -O OUTPUT_DIR -M MODEL_CHECKPOINT_PATH`

直接在文件里面改 不敲命令行


## Visualization

All log information generated during training will be stored in the `config["root_dir"]/<config_filename>/` directory. Assuming that the configuration file for training is `config/train/sample_16384.json`, the value of the` root_dir` parameter in `sample_16384.json` is` /home/UNet/`. Then, the logs generated during the current experimental training process will be stored In the `/home/UNet/sample_16384/` directory. The directory will contain the following:

- `logs/` directory: store Tensorboard related data, including loss curve, waveform file, speech file
- `checkpoints/` directory: stores all checkpoints of the model, from which you can restart training or speech enhancement
- `config.json` file: backup of the training configuration file

During the training process, we can use `tensorboard` to start a static front-end server to visualize the log data in the relevant directory:

```shell script

# For example, the "root_dir" parameter in the configuration file is "D:\yhc\lastdance", the configuration file name is "cbbrn1bwe16_48.json", and the default port is modified to 6000. The following commands can be used:
tensorboard --logdir D:\yhc\lastdance\cbbrn1bwe16_48 --port 6000
```


## Parameter Description

### Training

`config/train/<config_filename>.json`

The log information generated during the training process will be stored in`config["root_dir"]/<config_filename>/`.

```json5
{
    "seed": 0, // Random seeds to ensure experiment repeatability
    "description": "...",  // Experiment description, will be displayed in Tensorboard later
    "root_dir": "D:\\yhc\\lastdance", // Directory for storing experiment results
    "cudnn_deterministic": false,
    "trainer": { // For training process
        "module": "trainer.trainer", // Which trainer
        "main": "Trainer", // The concrete class of the trainer model
        "epochs": 1200, // Upper limit of training
        "save_checkpoint_interval": 10, // Save model breakpoint interval
        "validation":{
        "interval": 10, // validation interval
         "find_max": true, // When find_max is true, if the calculated metric is the known maximum value, it will cache another copy of the current round of model checkpoint.
        "custon": {
            "visualize_audio_limit": 20, // The interval of visual audio during validation. The reason for setting this parameter is that visual audio is slow
            "visualize_waveform_limit": 20, // The interval of the visualization waveform during validation. The reason for setting this parameter is because the visualization waveform is slow
            "visualize_spectrogram_limit": 20, // Verify the interval of the visualization spectrogram. This parameter is set because the visualization spectrum is slow
            "sample_length": 16384 // See train dataset
            } 
        }
    },
    "model": {
        "module": "model.unet_basic", // Model files used for training
        "main": "Model", // Concrete class of training model
        "args": {} // Parameters passed to the model class
    },
    "loss_function": {
        "module": "model.loss", // Model file of loss function
        "main": "mse_loss", // Concrete class of loss function
        "args": {} // Parameters passed to the model class
    },
    "optimizer": {
        "lr": 0.001,
        "beta1": 0.9,
        "beat2": 0.009
    },
    "train_dataset": {
        "module": "dataset.waveform_dataset", // Store the training set model file
        "main": "Dataset", // Concrete class of training dataset
        "args": { // The parameters passed to the training set class, see the specific training set class for details
            "dataset": "~/Datasets/SEGAN_Dataset/train_dataset.txt",
            "limit": null,
            "offset": 0,
            "sample_length": 16384,
            "mode":"train"
        }
    },
    "validation_dataset": {
        "module": "dataset.waveform_dataset",
        "main": "Dataset",
        "args": {
            "dataset": "~/Datasets/SEGAN_Dataset/test_dataset.txt",
            "limit": 400,
            "offset": 0,
            "mode":"validation"
        }
    },
    "train_dataloader": {
        "batch_size": 120,
        "num_workers": 40, // How many threads to start to preprocess the data
        "shuffle": true,
        "pin_memory":true
    }
}
```

### Enhancement

`config/enhancement/*.json`

```json5
{
    "model": {
        "module": "model.unet_basic",  // Store the model file
        "main": "UNet",  // The specific model class in the file
        "args": {}  // Parameters passed to the model class
    },
    "dataset": {
        "module": "dataset.waveform_dataset_enhancement",  // Store the enhancement dataset file
        "main": "WaveformDataset",  // Concrete class of enhacnement dataset
        "args": {  // The parameters passed to the dataset class, see the specific enhancement dataset class for details
            "dataset": "/home/imucs/tmp/UNet_and_Inpainting/data.txt",
            "limit": 400,
            "offset": 0,
            "sample_length": 16384
        }
    },
    "custom": {
        "sample_length": 16384
    }
}
```

During the enhancement, only the path of the noisy speech can be listed in the *.txt file, similar to this:

```text
# enhancement_*.txt

/home/imucs/tmp/UNet_and_Inpainting/0001_babble_-7dB_Clean.wav
/home/imucs/tmp/UNet_and_Inpainting/0001_babble_-7dB_Enhanced_Inpainting_200.wav
/home/imucs/tmp/UNet_and_Inpainting/0001_babble_-7dB_Enhanced_Inpainting_270.wav
/home/imucs/tmp/UNet_and_Inpainting/0001_babble_-7dB_Enhanced_UNet.wav
/home/imucs/tmp/UNet_and_Inpainting/0001_babble_-7dB_Mixture.wav
```

