# **Intelligent AI-Powered Five-Finger Gripper with EMG-Based Control**

This project aims to develop an advanced five-finger mechanical hand (gripper)
controlled by muscle activity detected from the forearm using electromyography
(EMG) sensors. The mechanical hand will be equipped with AI algorithms to
ensure safe and adaptive gripping of various objects without slippage. The
project includes hardware design, development of AI algorithms for object
manipulation, and integration of EMG sensors for precise control of hand
movements. Additionally, the hand should be capable of smooth object release.

## **Project Scope**

The goal of the DVA435 and CDT406 courses is to develop and evaluate AI
algorithms for electromyography (EMG) sensors, focusing on classifying three
distinct states: grip, rest, and release. These algorithms will be based on
empirical EMG measurements and will provide visual feedback to indicate the
detected state. The final versions must be optimized to run on an embedded
platform, such as the BeagleBone Green or an available NRF development card
from Nordic Semiconductor. The successful implementation of this project will
contribute to advancements in intelligent prosthetics and robotic grippers [1].

The students will conduct thorough testing of the AI algorithms by attempting to
classify EMG data corresponding to different grip states. The system's
performance will be evaluated based on accuracy, consistency, and
responsiveness in detecting these states.

It is crucial that all steps will be documented regarding the design process,
algorithms, hardware specifications, and testing outcomes. This includes:
  - Evaluations and implementation of AI algorithms
  - Integration with EMG sensors
  - Implementation on embedded platforms such as BeagleBone Green or Nordic Semiconductor's NRF52

## **Expected Outcomes**
  - Effective AI algorithms that accurately detect and classify grip, rest, and release states
  - Detailed documentation of design, development, and testing processes
  - Demonstration of AI algorithms' capabilities in a controlled environment

# **Useful stuff**
## **Setup** 
Based on how I set this up using Anaconda as Python interpreter. Will assume Anaconda installed and working. 

Step 1: Open command line inside project folder

Step 2: Install requirement.txt:
```
pip install -r requirements.txt
```


## **Folder Structure**
`/data/`: Contains all data. Raw data placed in `/data/raw/` and processed data in `/data/processed/`

`/logs/`: Contains any logs, such as training data, loss, validation stuff like this. Useful to have if we want to present graphs of the training in the final report. 

`/models/`: Here will any models that has been trained, are getting trained, be saved. 

`/notebooks/`: Any Jupyter notebooks that are part of this project goes in this folder. 

`/src/`: Main folder for any code. 
- `/src/data/`: Any coded related to the data. This includes loading and saving data, pre-processing and so on. 
- `/src/models/`: Any code related to creating, training and evaluating models goes here. 
- `/src/utils/`: Any code related to utility goes here. This includes path handling and other quality of life stuff. 
- `/src/visualizations/`: Any code related to plotting or showing any kind of data goes here. 

Feel free to add more folders as needed, just make sure to document them! 

[1] https://mdh.diva-portal.org/smash/record.jsf?pid=diva2%3A946321&dswid=-6351
