# EmotionsDetector

Emotions Detector based on FER2013 dataset


**Conda enviroment**
```
$ conda activate <env>
$ conda install pip
$ pip install -r requirements.txt
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
<img src="https://github.com/chacoff/EmotionsDetector/blob/main/readme/faces48x48.png" width="256">
<img src="https://github.com/chacoff/EmotionsDetector/blob/main/readme/databrief.png" width="360">
</p>

Originally, It has 7 classes, but we dropped it to 3 classes by grouping the nearest emotions:

<p align="left">
<img src="https://github.com/chacoff/EmotionsDetector/blob/main/readme/jup09_data.png" width="360">
</p>


**training**
```
$ jupyter notebook notebook-dir='c:/users/USER/'
>> training_SatAI_emotions1_3classes.ipynb
```

The model is on the repository compressed and splitted: _models/jup09_

<p align="left">
<img src="https://github.com/chacoff/EmotionsDetector/blob/main/readme/jup09_metrics.jpg" width="520">
</p>


**UI**

With the capability of capturing and recording webcams, videos and your screen. (IPcam(s) on the to-do list).

**parameters**
```xml
		<item name="theme">Reddit</item> <!-- cool themes -->
		<item name="halo">0.85</item> <!-- transparency effect -->
    <item name="CAM">NO</item> <!-- if YES the webcam/video will be capture, if NO, the screen-->
		<item name="VID">0</item> <!-- 0 for webcam, or video file address -->
		<item name="REC">YES</item> <!-- YES/NO to record webcam or the screen -->
		<item name="REC_file">jup09_3classes_1.avi</item> <!-- name of the file if the recording option is active -->
		<item name="FACTOR">2</item> <!-- factor to reduce the screen size to fit into the viewer -->
		
		<item name="modelsPath">models</item> <!-- -->
		<item name="protopath">deploy.prototxt</item> <!-- face detector proto file -->
		<item name="modelpath">res10_300x300_ssd_iter_140000.caffemodel</item> <!-- face detector -->
		<item name="threshold">0.45</item> <!-- to filter out weak face detections -->
		
		<item name="emotionsPath">jup09\</item> <!-- emotions classifier -->
		<item name="fps_xml">12</item> <!-- fps for recording -->
```

```
MoodyGUI.py
```
<p align="left">
<img src="https://github.com/chacoff/EmotionsDetector/blob/main/readme/moodyUI.png" width="600">
</p>

```
Emotions_UI_final.py
```
<p align="left">
<img src="https://github.com/chacoff/EmotionsDetector/blob/main/readme/Wfaces-1.gif" width="600">
</p>
