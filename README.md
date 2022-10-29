# MuSIC
MuSIC: Multi-Scale Image Classifier (A general-purpose deep neural framework for image classification)

A general-purpose Multi-Scale Image Classifier, named as MuSIC, is designed for script classification. MuSIC is an end-to-end multi-CNN parallel framework which consists of three main modules – (i) multi-scale CNN based prediction, (ii) a novel scale-wise weight computation module, and (iii) weight-aware decision for final prediction.

 ## MuSIC Architecture
![Overall architecture and working principle of MuSIC](https://user-images.githubusercontent.com/38031801/198822219-fb6eee8e-2bf5-45ad-99e6-e0600c8a1d4e.png)
                                                        <p align = "left">
                                                           Fig 1. Overall architecture and working principle of MuSIC
 </p>

## Salient Features of MuSIC
1. Multi-scale approach
2. Weight-aware decision mechanism to break tie situation in majority voting
3. General-purpose architecture
4. CPU and GPU enabled framework
5. Low memory usage
6. User defined scale selection
7. Flexibility in colour channel of input images

### Link to access MuSIC model
Access the file  Master file: https://github.com/iilabau/MuSIC/blob/master/usage_v2.py, Model: https://github.com/iilabau/MuSIC/blob/master/iilab.py

### Hardware and Software Requirements 
1. Install full Anaconda (python) distribution Package
2. GPU support: CUDA 10.1 version and  
                CudaDNN 7.6.5 library package
3. Processor: Core I5-8th Gen or above
4. Graphics Card: NVDIA GTX 1060
5. RAM: 8GB or above
6. Hard Disk: 512 GB or above

## MuSIC Evaluation
Model has been evaluated on multiple benchmark datasets and reported the obtained results.
### Datasets
1. Aliah University Text Non-text (AUTNT) (click [here](https://github.com/iilabau/AUTNTdataset) to download)
3. ICDAR CVSI-2015  (click [here](http://www.ict.griffith.edu.au/cvsi2015/Dataset.php) to download)
4. ICDAR 2019-MLT (wordlevel) (click [here](https://rrc.cvc.uab.es/?ch=15&com=introduction) to download)

### Results
#### AUTNT-Mixed

| No. of scales | P | R | F-M | Accuracy | Error-rate | FAR | FRR |
 --------------  ---  --- ---  ---------  -------------------------
|I, II,III,IV,V | 0.9716  |   |     |          |           |     |      |
|               |   |   |     |          |           |     |      |
|               |   |   |     |          |           |     |      |      



