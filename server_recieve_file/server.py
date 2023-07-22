from flask import Flask, session
from flask_socketio import SocketIO, emit
# import av
import wave
import time
import numpy as np
import pyaudio
import whisper
from flask_cors import CORS
from engineio.payload import Payload
import logging
import torch
import logging
# Set up Flask app and SocketIO
app = Flask(__name__)
CORS(app)
import threading


# Create a lock to ensure thread-safe access to frames_in list
frames_lock = threading.Lock()

# Function to perform transcription in a separate thread
def transcribe_frames(frames):
    with frames_lock:
        audio_data_np = np.frombuffer(b''.join(frames), np.int16).flatten().astype("float32") / 32768.0#int16
        # print(audio_data_np[:10],"????????????????????????????")

        audio_tensor = torch.from_numpy(audio_data_np)
        # print("======================================",audio_tensor,"============================================")
        print("LEngth audio tesnor",len(audio_tensor))
        transcription = model.transcribe(audio_tensor)
        print("***************************************************************"*10)
        print("Transcription is: ",transcription["text"])
        print("***************************************************************")
        socketio.emit('audio_dt', f' {transcription["text"]}  ')
        frames.clear()
# cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['SECRET_KEY'] = 'secret!'

Payload.max_decode_packets = 5000

socketio = socketio = SocketIO(app,  cors_allowed_origins="*")




# FORMAT = pyaudio.paFloat32
# FORMAT = pyaudio.paInt32
FORMAT = pyaudio.paInt16
CHANNELS = 1

frames_in = []
frames_out = []
RATE = 16000

CHUNK = 4096*10

audio = pyaudio.PyAudio()

print("Loading model...")
model = whisper.load_model("tiny.en")
print("Whisper Model Loaded")


def save_audio_buffer_to_wav(audio_buffer, filename, sample_rate=44100, channels=1, sample_width=2):
    # Convert the audio buffer to a numpy array
    audio_np = np.frombuffer(audio_buffer, dtype=np.int16)
    
    # Create a wave file with the specified parameters
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        
        # Write the audio data to the wave file
        wf.writeframes(audio_np.tobytes())
def save_wav(frames, filename, chunk = CHUNK  ,sample_format =FORMAT, channels = CHANNELS, fs = RATE):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    try:
        wf.setsampwidth(audio.get_sample_size(sample_format))
    except Exception as e:
        print(str(e))
        pass
    wf.setframerate(fs)
    time.sleep(0.1)
    wf.writeframes(b''.join(frames))
#     frames.clear()
    wf.close()
threshold = 63305.0
# threshold = 1499305.0
ms = 500
silence_duration = 0
last_speech_time = time.time()
start = False
def check_thresh(a):
    audio_buffer = np.frombuffer(a, dtype=np.int16)
    b = ((np.abs(np.fft.rfft(audio_buffer))) * 16000 / (len(a)//2)).mean()
    return b > threshold 
# Define SocketIO event handler for audio data
@socketio.on('audio_data', namespace='/')#, namespace='/')
def handle_audio_data(audio_data):
    global frames_in,silence_duration, last_speech_time,start
    # print("***********************",audio_data[:50])
    session['receive_count'] = session.get('receive_count', 0) + 1
    check = check_thresh(audio_data)

    # frames_in.append(audio_data)
    # print(check,">"*25)
    if check:
            # User is speaking
            last_speech_time = time.time()
            frames_in.append(audio_data)

            silence_duration = 0
            start =True
            print(f'\rUser is speaking: {len(frames_in)}', end="", flush=True)
    else:
            # User is silent
            silence_duration = time.time() - last_speech_time
            if start:
                frames_in.append(audio_data)
    # print(type(frames_in[0]),len(frames_in[0]),np.array(frames_in).shape,"****************")
    # print("\n",silence_duration,start,"???????????????????????")
    if silence_duration * 1000 >= ms and start:
        if len(frames_in)>=10:
            print("YEEEEEEEEEEEs")
            # save_wav(frames_in, 'audio.wav')
        #     audio_frame = av.AudioFrame.from_ndarray(np.frombuffer(audio_data, dtype=np.int16), format='s16')
            # print(len(frames_in))



            
             # Start a new thread for transcription
            transcription_thread = threading.Thread(target=transcribe_frames, args=(frames_in.copy(),))
            transcription_thread.start()
            # time.sleep(0.2)#0.2
            frames_in.clear()
            # socketio.emit('my_emittting', "server is emiiting to client")
        start = False

# Run Flask-SocketIO server
if __name__ == '__main__':
    frames_in = []  # Create a list to store audio frames
    print("Starting server...")
    socketio.run(app, host='0.0.0.0', port=5031)
    
 

   
    
 
