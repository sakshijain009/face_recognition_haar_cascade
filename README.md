# Face Recognition Using Opencv
Face Detection using Haar Cascades and OpenCV's LBPHFaceRecognizer algorithm.<br>
To run the project install opencv-contrib-python using pip:

```
pip install opencv-contrib-python
```
Clone the repo:
```
git clone https://github.com/sakshijain009/face_recognition_haar_cascade.git
```
To train the model, run the following command:
```
python faces_train.py
```
This will create the trained yaml file which will be use for recognition. To predict the images run the following command:
```
python face_recognition.py
```
