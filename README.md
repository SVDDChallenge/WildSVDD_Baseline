# WildSVDD_Baseline
Baseline systems for WildSVDD challenge @ MIREX 2024

[MIREX wiki page](https://www.music-ir.org/mirex/wiki/2024:Singing_Voice_Deepfake_Detection)
[SVDD challenge series webpage](https://main.singfake.org/)

## Baseline results

Please refer to the WildSVDD section in the SVDD 2024 @ SLT challenge overview paper. [https://arxiv.org/abs/2408.16132](https://arxiv.org/abs/2408.16132)

## Submission

Please send your score TXT files and a 2-page system description to [svddchallenge@gmail.com](mailto:svddchallenge@gmail.com). Feel free to submit as many systems as you want.

### Train the AASIST baseline with raw waveform frontend
```
python train.py --is_mixture True --frontend rawnet
```
### Evaluation and output the score file for submission
```
python eval.py -m ./logs/rawnet/20241001-153348/checkpoints/best_model.pt --encoder rawnet --is_mixture
```

