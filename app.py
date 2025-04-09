import streamlit as st
import joblib
import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import base64

# Function to normalize audio
def normalize_audio(y):
    max_amplitude = np.max(np.abs(y))
    return y / max_amplitude

# Function to extract features from the audio file
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

# Function to load and test the model
def test_model(test_audio_path):
    model_path = 'audio_classification_svm.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError("The pre-trained model file does not exist. Please upload the model file.")

    model = joblib.load(model_path)

    # Extract features from the test audio (WAV format)
    features = extract_features(test_audio_path)

    # Predict using the trained SVM model
    prediction = model.predict([features])

    # Return result
    return "REAL" if prediction == 1 else "DEEPFAKE"

# Function to plot audio waveform
def plot_waveform(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    y = normalize_audio(y)

    # Plot the waveform
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

# Function to plot Mel Spectrogram
def plot_mel_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    y = normalize_audio(y)

    # Compute Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)

    # Convert to decibels
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Plot the Mel Spectrogram
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title("Mel Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    st.pyplot(fig)

# Streamlit front-end
st.title("AudioVeritas")
st.markdown("""
This application allows you to upload an audio file (WAV format) and determine if the audio is *REAL* or *DEEPFAKE* using a trained SVM model.
""")

# Encode the background video into base64
video_file = open("IMG_6004.MP4", "rb").read()
encoded_video = base64.b64encode(video_file).decode()

# Add background video
st.markdown(f"""
<style>
.video-container {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
}}
.stApp {{
    background: transparent;
}}
</style>
<div class="video-container">
    <video width="100%" height="100%" autoplay loop muted>
        <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
    </video>
</div>
""", unsafe_allow_html=True)

# Check if pre-trained model exists or prompt for upload
model_path = 'audio_classification_svm.pkl'
if not os.path.exists(model_path):
    st.warning("The pre-trained model file is missing. Please upload the model file.")
    uploaded_model_file = st.file_uploader("Upload the pre-trained model (.pkl file)", type="pkl")
    if uploaded_model_file is not None:
        with open(model_path, "wb") as f:
            f.write(uploaded_model_file.getbuffer())
        st.success("Model file uploaded successfully! You can now classify audio files.")

# File upload and prompts
uploaded_file = st.file_uploader("Choose an audio file", type="wav")
if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the audio file
    st.audio(uploaded_file, format="audio/wav")

    # Button to classify the audio
    if st.button("Classify"):
        try:
            # Get the result from the model
            result = test_model(temp_path)

            # Display the result
            st.subheader("Result")
            st.write(f"The audio is: *{result}*")

            # Plot waveform
            st.subheader("Audio Waveform")
            plot_waveform(temp_path)

            # Display the Mel Spectrogram visualization
            st.subheader("Mel Spectrogram Visualization")
            plot_mel_spectrogram(temp_path)
        except FileNotFoundError as e:
            st.error(str(e))
        finally:
            # Clean up the temporary file after classification
            if os.path.exists(temp_path):
                os.remove(temp_path)
else:
    st.warning("Please upload an audio file to begin.")