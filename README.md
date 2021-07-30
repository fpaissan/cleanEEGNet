---

<div align="center">    
 
# Towards a Domain-specific Neural Network Approach for EEG Bad Channel Detection

</div>
 
## Abstract
Electroencephalogram (EEG) is prone to several artifacts that often leads to misclassification of neural features in Brain-Computer Interfaces (BCI). Traditionally, detecting and removing bad EEG electrodes (or channels) is often the first and most critical step in cleaning the data. There are a few automated tools, and each uses its own statistical signal processing techniques with tunable hyperparameters (e.g., the z-score threshold for amplitude-based outlier detection). To the best of our knowledge, an objective deep learning approach for this specific problem is still missing. This paper proposes _cleanEEGNet_, a Convolutional Neural Network to identify the bad channels in EEG signals. We carefully chose the model hyperparameters (i.e., kernel size and stride) to mimic the conventional detection of bad channels performed via visual inspection. An open source dataset from [OpenNeuro](https://openneuro.org/datasets/ds002034/versions/1.0.1) with annotated bad channels is used to train and validate the network.For a benchmark comparison, we chose four state-of-the-art automated conventional methods for bad channel removal, including FASTER and HAPPE. %Of the considered traditional methods,
Among them, HAPPE performed the best, achieving a balanced accuracy of 66%, while _cleanEEGNet_ outperformed HAPPE by 17% with a balanced accuracy of 78\%.

#### Example: cifar10 
To train PhiNets for cifar10, you can run:

```
python __main__.py cifar10 data
```

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
