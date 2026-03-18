CNN Finger Counter

A convolutional neural network that classifies how many fingers you're holding up (0-5) in real time using your webcam.
I built this to learn CNNs properly. Instead of following a tutorial with someone else's dataset, I collected my own images of people holding up different numbers of fingers and trained a model from scratch. Then I wrote a webcam script that loads the model and predicts live.
Demo

<img width="638" height="514" alt="1_finger" src="https://github.com/user-attachments/assets/33e3a854-f656-4ca4-87ce-7f9bad2fdf7b" />
<img width="641" height="511" alt="2_fingers" src="https://github.com/user-attachments/assets/e39f00f8-70b0-4ac0-8076-65d1b9624ef0" />
<img width="639" height="510" alt="3_fingers" src="https://github.com/user-attachments/assets/e169b810-3329-46ec-a6c6-d49b226fe37b" />

How It Works

Collected ~570 images across 5 classes (0-4 fingers), mostly from Google Images
Cleaned the dataset, removing corrupt files and non-image junk
Built a CNN using TensorFlow/Keras with 3 Conv2D layers, MaxPooling, and a Dense softmax output
Trained for 20 epochs with a 70/20/10 train/val/test split
Wrote a separate Python script that loads the trained model and runs predictions on a live webcam feed using OpenCV

Files

CNN_build_v001.ipynb - Training notebook. Data loading, preprocessing, model building, training, evaluation
finger_counter_cam.py - Webcam script. Loads the saved model and predicts in real time
data/ - Training images organised by class folder
models/ - Not included (168MB). Run the notebook to generate the model

Model Architecture

Conv2D(16, 3x3) + MaxPooling
Conv2D(32, 3x3) + MaxPooling
Conv2D(16, 3x3)
Flatten
Dense(256, relu)
Dense(5, softmax)

Usage
Train the model:
Open CNN_build_v001.ipynb and run all cells
Run the webcam:
python finger_counter_cam.py
Press q to quit.
What I'd Improve

More training data. 570 images across 5 classes is small
Better dataset quality. Some images are stock photos with watermarks
Add class for 5 fingers (currently 0-4)
Try transfer learning with a pretrained model like MobileNet

Built With
Python, TensorFlow/Keras, OpenCV, NumPy, Matplotlib
