import librosa
import soundfile
import os, glob, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as matplt
from matplotlib import rcParams
import seaborn as sns

from sklearn.metrics import classification_report,ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_score, recall_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

observed_emotions=['calm', 'happy', 'fearful', 'disgust']

def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("C:/Users/kumar/OneDrive/Desktop/Research_Project_PP1/Dataset/Actor_*/*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


x_train,x_test,y_train,y_test=load_data(test_size=0.25)
print(x_train)
print("\n")
print(x_test)

print((x_train.shape[0], x_test.shape[0]))

print(f'Features extracted: {x_train.shape[1]}')


random_model = RandomForestClassifier()

random_model.fit(x_train, y_train)
y_pred = random_model.predict(x_test)

r_accuracy = random_model.score(x_test, y_test)
cm = confusion_matrix(y_test, y_pred)

print('Presentage of Random Forest Classifier scores')
print(f'Random Forest Classifier Model accuracy\t: {r_accuracy}')
print(f'Presentage\t: {"{:.1%}".format(r_accuracy)}')
print(classification_report(y_test, y_pred))


model = SVC()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

s_accuracy = model.score(x_test, y_test)
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

print('Presentage of Support vector Classifier scores')
print(f' Support vector Classifier Model accuracy\t: {s_accuracy}')
print(f'Presentage\t: {"{:.1%}".format(s_accuracy)}')
print(classification_report(y_test, y_pred))

print('Confusion Matrix - ')
print(cm)





model = DecisionTreeClassifier()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

d_accuracy = model.score(x_test, y_test)
cm = confusion_matrix(y_test, y_pred)

print('Decision Tree scores')
print(f'Decision Tree Model accuracy\t: {d_accuracy}')
print(f'Presentage\t: {"{:.1%}".format(d_accuracy)}')
print(classification_report(y_test, y_pred))

print('Confusion Matrix - ')
print(confusion_matrix(y_test, y_pred))

print('Confusion Matrix - ')
print(cm)



#Initialize the Multi Layer Perceptron Classifier
mlp_model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
#Training the model using training data
mlp_model.fit(x_train, y_train)
y_pred = mlp_model.predict(x_test)
# Testing the model using the testing data
mlp_accuracy = mlp_model.score(x_test, y_test)
cm = confusion_matrix(y_test, y_pred)

print('Presentage of MLP Classifier scores')
print(f'MLP Classifier Model accuracy\t: {mlp_accuracy}')
print(f'Presentage\t: {"{:.1%}".format(mlp_accuracy)}')
print(classification_report(y_test, y_pred))


#Compare Algorithum
matplt.bar(['Decision Tree','Random Forest','Support Vector Machine', 'MLP Classification'],[d_accuracy,r_accuracy,s_accuracy, mlp_accuracy])
matplt.xlabel("Algorithm Type")
matplt.ylabel("Model Accuracy")
matplt.show()