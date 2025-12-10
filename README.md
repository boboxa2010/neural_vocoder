# HiFi-GAN with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

PyTorch implementation of HiFi-GAN.

See the task assignment [here](https://github.com/markovka17/dla/tree/2025/hw3_nv).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/boboxa2010/neural_vocoder.git
   cd neural_vocoder
   ```

2. Create conda environment (strongly recommended):
   ```bash
   conda create -n dla python=3.10
   conda activate dla
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

### Train model

To train a model, run the following command:

```bash
$ python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run e2e synthesis:

```bash
$ python synthesize.py \
        synthesizer.from_pretrained=checkpoints/model.pth \
        datasets=custom_dir_dataset \
        datasets.inference.path=data/datasets/custom_dataset
```

To run e2e synthesis on single utterance:

```bash
$ python synthesize.py \
        synthesizer.from_pretrained=checkpoints/model.pth \
        "synthesizer.text='I am MOP and I love Yandex'" \
        synthesizer.output_name="test_utterance"
```

To run vocoder on single utterance:

```bash
$ python synthesize.py \
        synthesizer.from_pretrained=checkpoints/model.pth \
        datasets=custom_audio_dir_dataset \
        datasets.inference.path=data/datasets/custom_dataset \
        +synthesizer.resynthesize=True
```

The results will be saved in next path:
` ROOT_PATH/data/saved/{synthezer.save_path}`


All the models configs can be found in `src/configs/model`.


## How to load checkpoint

To load the final checkpoint, run the following command:

```bash
$ ./download_yadisk.sh https://disk.yandex.ru/d/B9B_NgMlQgu2xw checkpoints/model.pth
```

## How to load your dataset

Your dataset should be placed on yandex disk and has the next structure:
```
NameOfTheDirectoryWithUtterances
└── transcriptions
    ├── UtteranceID1.txt
    ├── UtteranceID2.txt
    .
    .
    .
    └── UtteranceIDn.txt
```


```bash
$ ./download_dataset.sh dataset_link_on_yadisk content/datasets/custom_dataset
```

## How to reproduce the best result

```bash
$ python3 train.py -cn=hifi_gan_v1 datasets.train.chunk_size=8192 datasets.val.chunk_size=8192 +datasets.train.padding_mode=reflect +datasets.val.padding_mode=reflect writer.run_name=hifi_gan_chunk_8192_reflect_long trainer.n_epochs=250
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
