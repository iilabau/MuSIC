# MuSIC (Multi-scale Image Classifier)
 
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
#### Input parameters of master file (usage_V2.py):
	1. Dimension of scales
	2. Path for training data
	3. Path for test data.
####  Model file (iilab.py) contains both MuSIC Version 2 and version 1:
	1. MuSIC version 2 Model (run class musicv2)
	2. MuSIC version 1 Model (run class MuSIC)

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
#### Results on AUTNT-Mixed Dataset
|No. of scale|Precision |Recall |F-Measure |Accuracy |Error-rate |FAR |FRR |
|:-----------|:-|:-|:---|:--------|:----------|:---|:---|
|I, II, III, IV, V|0.9716|0.9742|0.9729|0.9746|0.0254|-|-|
|I, II, III, IV, V, VI, VII|0.9763|	0.9799|	0.9780|	0.9803|	0.0197|	0.0097|	0.0199|
|I, II, III, IV, V, VI, VII, VIII, IX|0.9722|	0.9765|	0.9742|	0.9771|	0.0229|	-|	-|
|average|0.9733|	0.9768|	0.9750|	0.9773|	0.0226|	-|	-|

#### Results on ICDAR CVSI-2015 Dataset
|No. of scale|Precision |Recall |F-Measure |Accuracy |Error-rate |FAR |FRR |
|:-----------|:-|:-|:---|:--------|:----------|:---|:---|
|I, II, III, IV, V|0.9257	|0.9295|	0.9252	|0.9251|	0.0749|	-|	-|
|I, II, III, IV, V, VI, VII|0.9308|	0.9348|	0.9305|	0.9301|	0.0699|	-|	-|
|I, II, III, IV, V, VI, VII, VIII, IX|0.9316|	0.9344|	0.9313|	0.9310	|0.0690	|0.0076|	0.0635|
|average|0.9293|	0.9329	|0.9290	|0.9287|	0.0712|	-|	-|

#### Results on ICDAR 2019-MLT (wordlevel) Dataset
|No. of scale|Precision |Recall |F-Measure |Accuracy |Error-rate |FAR |FRR |
|:-----------|:-|:-|:---|:--------|:----------|:---|:---|
|I, II, III, IV, V|0.8443|	0.9782|	0.8973|	0.9634|	0.0366|	-|	-|
|I, II, III, IV, V, VI, VII|0.8783|	0.9872|	0.9246|	0.9721|	0.0279	|0.0293|	0.0127|
|I, II, III, IV, V, VI, VII, VIII, IX|0.8804|	0.9880|	0.9272|	0.9706|	0.0294|	-|	-|
|average|0.8676|	0.9844|	0.9163|	0.9687|	0.0313|	-|	-|

	 

#### Performance Comparison between individual scale and multi-scale driven MuSIC model Across Multiple Datasets
![image](https://user-images.githubusercontent.com/38031801/198828013-ae514188-0e92-4c76-b3bd-1902ffbd46e6.png)
<p align="left">
   Fig 2. Performance comparison between single-scale and multi-scale driven MuSIC framework on five benchmark datasets. 
          Average classification accuracies of nine different  scales are considered for performance assessment. 
          Moreover, stability is measured using standard deviation and mean accuracy.
</p>

### Relevant Paper
In future the following paper may be consulted and cited while referring to this dataset (upon acceptance)
> **Khan, T., Saif, M. and Mollah, A.F., 2024. MuSIC: A Novel Multi-Scale Deep Neural Framework for Script Identification in the Wild.**

### Contributors
- Dr. Ayatullah Faruk Mollah, Assistant Professor, Department of Computer Science and Engineering, Aliah University, Kolkata, India
- Dr. Tauseef Khan, Assistant Professor, School of Computer Science and Engineering, VIT-AP University, Amaravati, Andhra Pradesh.

### Contact
Dr. Ayatullah Faruk Mollah, Email: afmollah@alaih.ac.in, 
Dr. Tauseef Khan, Email: tauseef.hit2013@gmail.com

