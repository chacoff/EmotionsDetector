# EmotionsDetector

Emotions Detector based on FER2013 dataset


**Conda enviroment**
```
$ conda activate <env>
$ conda install pip
$ pip freeze > requirements.txt
```
**PIP enviroment**
```
$ python3 -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```

Pay attention with the CUDA drivers: Build cuda_11.3.r11.3/compiler.29745058_0
```
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Sun_Mar_21_19:24:09_Pacific_Daylight_Time_2021
Cuda compilation tools, release 11.3, V11.3.58
Build cuda_11.3.r11.3/compiler.29745058_0
```


**git**
```
$ git clone https://github.com/chacoff/EmotionsDetector
```


**dataset**

it can be download from kaggle: https://datarepository.wolframcloud.com/resources/FER-2013

<p align="left">
<img src="https://github.com/chacoff/EmotionsDetector/blob/main/models/faces48x48.png" width="256">
<img src="https://github.com/chacoff/EmotionsDetector/blob/main/models/databrief.png" width="360">
</p>
