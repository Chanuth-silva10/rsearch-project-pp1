import librosa
import soundfile
import os
import glob
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pyaudio
import wave


emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
1

observed_emotions = ["happy", "calm"]


model = MLPClassifier(
    alpha=0.01,
    batch_size=256,
    epsilon=1e-08,
    hidden_layer_sizes=(300,),
    learning_rate="adaptive",
    max_iter=500,
)


def recordAudio():
    chunk = 1024  
    sample_format = pyaudio.paInt16  
    channels = 1
    fs = 48100 
    seconds = 5
    filename = "RealTimeRecordAudio/Predict-Record-Audio.wav"

    p = pyaudio.PyAudio()

    print("Recording")

    stream = p.open(
        format=sample_format,
        channels=channels,
        rate=fs,
        frames_per_buffer=chunk,
        input=True,
    )

    frames = [] 

    
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

   
    stream.stop_stream()
    stream.close()
    
    p.terminate()

    print("Finished recording")

  
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b"".join(frames))
    wf.close()


def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(
                librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0
            )
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(
                librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0
            )
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result


def load_data():
    x, y = [], []
    for file in glob.glob(
        "C:/Users/kumar/OneDrive/Desktop/Research_Project_PP1/Dataset/Actor_*/*.wav"
    ):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]

        

        if emotion in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return np.array(x), y


def trainModel():
    
    x, y = load_data()

    # Normalize features
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    
    x_train, x_test, y_train, y_test = train_test_split( x, y ,test_size=0.25, random_state=9)
    
    
    print((x_train.shape[0], x_test.shape[0]))

    print(f"Features extracted: {x_train.shape[1]}")

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

    print("Accuracy: {:.2f}%".format(accuracy * 100))


def record_predictAudio():
    x_predictAudio = []
    recordAudio()  
    file = "C:/Users/kumar/OneDrive/Desktop/Research_Project_PP1/RealTimeRecordAudio/Predict-Record-Audio.wav"  # Recorded audio filepath
    print("" + file)
    featurePredictAudio = extract_feature(
        file, mfcc=True, chroma=True, mel=True
    ) 
    x_predictAudio.append(featurePredictAudio)
    y_predictAudio = model.predict(np.array(x_predictAudio))
    print("Emotion Predicted: {}".format(y_predictAudio))


def predictAudio():
    file = input("Please enter path to your file.\n")
    x_predictAudio = []
    featurePredictAudio = extract_feature(
        file, mfcc=True, chroma=True, mel=True
    )  
    x_predictAudio.append(featurePredictAudio)
    y_predictAudio = model.predict(np.array(x_predictAudio))
    print("Emotion Predicted: {}".format(y_predictAudio))


while True:
    choice = int(
        input(
            "Enter 1 to create and train model. \nEnter 2 to record and predict audio. \nEnter 3 to predict on pre-recorded audio. \nEnter 4 to quit. \n"
        )
    )
    if choice == 1:
        trainModel()
    elif choice == 2:
        record_predictAudio()
    elif choice == 3:
        predictAudio()
    else:
        quit()
