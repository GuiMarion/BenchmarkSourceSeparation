# BenchmarkSourceSeparation
Project for a class validation at ATIAM master's degree at IRCAM on a survey of source separation on self-made studio recording.

This project is made with the supervision of Umut Şimşekli and Benoit Fabre and realised by 
  - Valentin Bilot
  - Gabriel Dias Neto
  - Clement Le-Moine
  - Guilhem Marion
  - Yann Teytaut

## Presentation

This project presents a quick survey of the problem of source separation. To this end, we will investigate three state-of-the-art signal processing algorithms and compare them for remixing purpose with two means: objective measures and perceptive experiment. We show that these results are not correlated and that low- separated sources with low artifacts tend to be better perceived than high-separated with artifacts sources.

## Usage

### Extraction
The script extract.py in audio/ extracts files in order to fill the experiment script.

### Experiment

The script MainExperiment launch the experiment, you can change parameter by editing the file.
    
    Usage: Python3 MainExperiment.py <ROOM or ANECHOIC>

### Database

The scripts anal_ROOM.py and anal_ANECHOIC.py allow to fill the database and compute statistical tests onto.

    Usage: anal_ROOM.py [options] <subject to fill>

    Options:
      -h, --help            show this help message and exit
      -d NAME, --name=NAME  Name of the database.
      -p PPRINT, --print=PPRINT
                            1 if you want to print the data.
## Licence 

This project entirely open source, and may by used for any purpose.
