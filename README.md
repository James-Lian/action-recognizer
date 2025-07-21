An action recognizer framework built with Python, Mediapipe, Tensorflow, and OpenCV. Takes the user's joint positions and predicts the user's actions from a webcam based on training data.

## How it works
Using the Mediapipe and OpenCV packages, the user can gather training data by recording their own movements. Then, by training a custom Keras LSTM, the user can then train a machine learning model to recognize their motions and attribute them to a labelled class.

![GIF of learning model recognizing user movements from camera](https://github.com/James-Lian/action-recognizer/blob/main/ActionRecognition.gif)

See example for additional scripts on how to use. The data-gathering.py script makes use of an Arduino IR remote to control the script. If you do not have one, you will need to edit the script to your case. Currently used to built a Breath of the Wild exercise simulator: https://github.com/James-Lian/botw-exercise-simulator
