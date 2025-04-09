import os
import numpy as np
import librosa
from pydub import AudioSegment
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib  # For saving and loading the model
from tqdm import tqdm  # For the progress bar
import time  # For time tracking

# Specify the path to ffmpeg and ffprobe if they're not found automatically
#AudioSegment.ffmpeg = 'C:/ffmpeg/ffmpeg-7.1-essentials_build/bin/ffmpeg.exe'   Replace with your actual FFmpeg path
#AudioSegment.ffprobe = 'C:/ffmpeg/ffmpeg-7.1-essentials_build/bin/ffprobe.exe'   Replace with your actual ffprobe path

# Normalize audio
def normalize_audio(y):
    max_amplitude = np.max(np.abs(y))
    return y / max_amplitude

# Feature extraction function
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None, res_type='kaiser_best')
    y = normalize_audio(y)
    n_fft = 2048
    hop_length = 512

    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    # Aggregate features into a single vector
    features = np.hstack([
        mfccs.mean(axis=1),
        chroma.mean(axis=1),
        spectral_centroid.mean(),
        spectral_bandwidth.mean(),
        zcr.mean()
    ])
    return features

# Function to train the SVM model with class weights
def train_model(data_dir):
    features = []
    labels = []

    # Start time tracking
    start_time = time.time()

    # Load audio files and extract features with progress bar
    print("Extracting features from the audio files...")
    for label, folder in [('original', 1), ('deepfake', 0)]:
        folder_path = os.path.join(data_dir, label)
        files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]  # Only process WAV files
        # Add progress bar for feature extraction
        for file in tqdm(files, desc=f"Processing {label} files", unit="file"):
            wav_path = os.path.join(folder_path, file)
            features.append(extract_features(wav_path))
            labels.append(folder)

    # Convert to NumPy arrays
    features = np.array(features)
    labels = np.array(labels)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train an SVM model with class weights to handle imbalance
    print("Training the SVM model...")
    model = SVC(kernel='linear', probability=True, class_weight='balanced')  # Added class_weight='balanced'
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Save the trained model
    joblib.dump(model, 'audio_classification_svm.pkl')
    print("Model saved as 'audio_classification_svm.pkl'")

    # End time tracking and display total time taken
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")

# Function to load and test the model
def test_model(test_audio_path):
    model = joblib.load('audio_classification_svm.pkl')
    
    # Extract features from the test audio (WAV format)
    features = extract_features(test_audio_path)

    # Predict using the trained SVM model
    prediction = model.predict([features])
    
    # Output the result
    if prediction == 1:
        print("The audio is REAL.")
    else:
        print("The audio is DEEPFAKE.")

# Main function
def main():
    # Check if the model already exists
    if not os.path.exists('audio_classification_svm.pkl'):
        print("Model not found. Training the model...")
        data_dir = input("Enter the path to the dataset directory (contains 'original' and 'deepfake' folders): ")
        if os.path.exists(data_dir):
            train_model(data_dir)
        else:
            print(f"Dataset directory '{data_dir}' not found. Please ensure it's correct.")
            return
    else:
        print("Model found. You can directly test the model.")
    
    # Testing phase (after training is done or if the model already exists)
    test_audio_path = input("Enter the path to the test audio file (WAV format): ")
    if os.path.exists(test_audio_path):
        test_model(test_audio_path)
    else:
        print("Test audio file not found.")
        return

if __name__ == "_main_":
    main()