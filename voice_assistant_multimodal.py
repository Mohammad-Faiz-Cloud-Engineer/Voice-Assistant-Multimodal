import os
import sounddevice as sd
import numpy as np
import torch
import whisper
from TTS.api import TTS
from termcolor import colored
import pyfiglet
import soundfile as sf
import requests
import threading
import time
import keyboard
import cv2
import json
import base64
import pyautogui  # For screenshots
import serial
import time

# --- Arduino Servo Setup ---
try:
    arduino = serial.Serial('COM3', 9600, timeout=1)
    time.sleep(2)
    print("Connected to Arduino on COM3")
except Exception as e:
    print("Arduino not connected:", e)
    arduino = None


# ===============================
# Helper UI
# ===============================
def banner(text, color="cyan"):
    print(colored(pyfiglet.figlet_format(text, font="slant"), color))

def section(title, prefix=">>"):
    print(colored("\n" + "=" * 60, "yellow"))
    print(colored(f" {prefix}  {title}", "green", attrs=["bold"]))
    print(colored("=" * 60, "yellow"))

def ask_choice(prompt, options):
    while True:
        choice = input(colored(f"{prompt} ({'/'.join(options.keys())}): ", "yellow")).strip().upper()
        if choice in options:
            return choice
        print(colored("Invalid choice, try again!", "red"))

# ===============================
# Config
# ===============================
STOP_PHRASE = "stop listening"
CAMERA_INDEX = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
print(colored(f"Using device: {device}\n", "magenta", attrs=["bold"]))

text_model = "qwen2-vl-7b-instruct"
vision_model = "qwen2-vl-7b-instruct"

# ===============================
# Startup
# ===============================
banner("COQUI PIPELINE", "magenta")
print(colored("Would you like to:", "cyan"))
print(colored("1. Auto-load instantly (default voice, no setup)", "yellow"))
print(colored("2. Try advanced setup (custom voice, cloning, emotions, language)", "yellow"))
mode_choice = input(colored("Enter 1 or 2: ", "cyan")).strip()
AUTO_MODE = mode_choice == "1"

# ===============================
# Help Section
# ===============================
section("Voice Command Help", ">>")
print("""Say any of the following:
Camera Control - "Turn camera on" / "Turn camera off" - "Take picture" / "Snap photo" - "Record a short video" 
Screen Control - "Take screenshot" / "Capture screen" 
Vision and Object Understanding - "Describe image" / "Analyze image" - "Describe screenshot" / "Analyze screenshot" 
Object & Person Detection - "Find car in image" - "Detect faces" / "Detect body" 
Control - "Stop listening" – exits the program
""")

# ===============================
# Audio Utilities
# ===============================
def record_audio(duration=5, fs=16000):
    print(colored("Mic active... speak now!\n", "cyan", attrs=["bold"]))
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    return np.squeeze(audio)

pause_flag = False
stop_flag = False

def play_audio(file_path):
    global pause_flag, stop_flag
    data, fs = sf.read(file_path, dtype="float32")
    stop_flag = False
    pause_flag = False

    def audio_thread():
        global pause_flag, stop_flag
        stream = sd.OutputStream(samplerate=fs, channels=data.shape[1] if data.ndim > 1 else 1)
        stream.start()
        i = 0
        blocksize = 1024
        while i < len(data):
            if stop_flag:
                break
            if pause_flag:
                time.sleep(0.1)
                continue
            end = i + blocksize
            stream.write(data[i:end])
            i = end
        try:
            stream.stop()
        except Exception:
            pass
        stream.close()

    t = threading.Thread(target=audio_thread)
    t.start()
    print("Playing... Press 'p' to pause/resume, 's' to stop.")
    while t.is_alive():
        if keyboard.is_pressed("p"):
            pause_flag = not pause_flag
            print("Paused" if pause_flag else "Resumed")
            time.sleep(0.3)
        elif keyboard.is_pressed("s"):
            stop_flag = True
            print("Stopped")
            break
        time.sleep(0.1)
    t.join()

# ===============================
# Load Whisper + Coqui
# ===============================
section("Loading Whisper model", ">>")
whisper_model = whisper.load_model("small")
print(colored("Whisper ready!\n", "cyan", attrs=["bold"]))

if AUTO_MODE:
    section("Auto Mode", ">>")
    selected_voice = "tts_models/en/ljspeech/tacotron2-DDC"
    tts = TTS(selected_voice, progress_bar=False, gpu=(device == "cuda"))
    tts.extra_args = {}
    print(colored("Auto-load complete (default Coqui voice active)", "green"))
else:
    # Advanced setup
    banner("VOICE SETUP", "cyan")
    voice_choice = ask_choice("Keep the default voice?", {"Y": "Yes", "N": "No"})
    if voice_choice == "N":
        available_voices = [
            "tts_models/en/ljspeech/tacotron2-DDC",
            "tts_models/en/ljspeech/glow-tts",
            "tts_models/en/vctk/vits",
            "tts_models/multilingual/multi-dataset/your_tts"
        ]
        for idx, v in enumerate(available_voices, 1):
            print(colored(f"  {idx}. {v}", "cyan"))
        while True:
            try:
                sel = int(input(colored("\nSelect a voice number: ", "yellow")))
                if 1 <= sel <= len(available_voices):
                    selected_voice = available_voices[sel - 1]
                    print(colored(f"Selected voice: {selected_voice}", "green"))
                    break
            except ValueError:
                pass
            print(colored("Invalid selection, try again!", "red"))
    else:
        selected_voice = "tts_models/en/ljspeech/tacotron2-DDC"

    section("Loading Coqui TTS", ">>")
    tts = TTS(selected_voice, progress_bar=False, gpu=(device == "cuda"))
    print(colored("Coqui TTS ready!\n", "cyan", attrs=["bold"]))

    selected_speaker = None
    selected_emotion = None
    selected_language = None
    speaker_wav = None

    # Multi-speaker
    if getattr(tts, "speakers", None):
        section("Multi-Speaker Model Detected", ">>")
        for i, spk in enumerate(tts.speakers, 1):
            print(colored(f" {i}. {spk}", "green"))
        while True:
            try:
                choice = int(input(colored("\nSelect speaker number: ", "yellow")))
                if 1 <= choice <= len(tts.speakers):
                    selected_speaker = tts.speakers[choice - 1]
                    print(colored(f"Selected speaker: {selected_speaker}", "green"))
                    break
            except ValueError:
                pass
            print(colored("Invalid selection, try again!", "red"))

    # Multilingual
    if getattr(tts, "languages", None):
        section("Multi-Lingual Model Detected", ">>")
        for i, lang in enumerate(tts.languages, 1):
            print(colored(f" {i}. {lang}", "green"))
        while True:
            try:
                choice = int(input(colored("\nSelect language number: ", "yellow")))
                if 1 <= choice <= len(tts.languages):
                    selected_language = tts.languages[choice - 1]
                    print(colored(f"Selected language: {selected_language}", "green"))
                    break
            except ValueError:
                pass
            print(colored("Invalid selection, try again!", "red"))

    # Emotions
    emotions = ["Neutral", "Happy", "Sad", "Angry", "Excited"]
    section("Voice Emotions", ">>")
    emotion_choice = ask_choice("Would you like to add an emotion?", {"Y": "Yes", "N": "No"})
    if emotion_choice == "Y":
        for i, emo in enumerate(emotions, 1):
            print(colored(f" {i}. {emo}", "green"))
        while True:
            try:
                choice = int(input(colored("\nSelect emotion number: ", "yellow")))
                if 1 <= choice <= len(emotions):
                    selected_emotion = emotions[choice - 1]
                    print(colored(f"Selected emotion: {selected_emotion}", "green"))
                    break
            except ValueError:
                pass
            print(colored("Invalid selection, try again!", "red"))

    # Cloning
    section("Voice Cloning", ">>")
    clone_choice = ask_choice("Would you like to clone a voice?", {"Y": "Yes", "N": "No"})
    if clone_choice == "Y":
        print(colored("Enter path to your 6-sec .wav file: ", "yellow"), end="")
        speaker_wav = input().strip()

    # Combine optional params
    kwargs = {}
    if selected_emotion: kwargs["emotion"] = selected_emotion
    if selected_speaker: kwargs["speaker"] = selected_speaker
    if selected_language: kwargs["language"] = selected_language
    if speaker_wav: kwargs["speaker_wav"] = speaker_wav

    tts.extra_args = kwargs
    print(colored("Voice setup complete!\n", "green", attrs=["bold"]))

# ===============================
# Camera + Screenshot
# ===============================
camera = None
preview_thread = None
preview_running = False

def camera_preview():
    global camera, preview_running
    while preview_running and camera is not None:
        ret, frame = camera.read()
        if not ret:
            break
        cv2.imshow("Camera Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    preview_running = False
    cv2.destroyAllWindows()

def turn_camera_on():
    global camera, preview_thread, preview_running
    if camera is None:
        camera = cv2.VideoCapture(CAMERA_INDEX)
        if not camera.isOpened():
            print(colored("Failed to open camera.", "red"))
            camera = None
            return False
        preview_running = True
        preview_thread = threading.Thread(target=camera_preview, daemon=True)
        preview_thread.start()
        print(colored("Camera turned ON (preview active).", "green"))
        return True
    else:
        print(colored("Camera already ON.", "yellow"))
        return True

def turn_camera_off():
    global camera, preview_running
    if camera:
        preview_running = False
        time.sleep(0.5)
        try:
            camera.release()
        except Exception:
            pass
        camera = None
        cv2.destroyAllWindows()
        print(colored("Camera turned OFF.", "yellow"))
    else:
        print(colored("Camera already OFF.", "yellow"))

def take_picture():
    global camera
    if camera is None:
        print(colored("Camera is OFF. Turn it ON first.", "yellow"))
        return None
    ret, frame = camera.read()
    if not ret:
        print(colored("Failed to capture image.", "red"))
        return None
    image_path = "captured.jpg"
    cv2.imwrite(image_path, frame)
    print(colored(f"Image saved as {image_path}", "green"))
    return image_path

def take_screenshot():
    image_path = "screenshot.jpg"
    screenshot = pyautogui.screenshot()
    screenshot.save(image_path)
    print(colored(f"Screenshot saved as {image_path}", "green"))
    return image_path

import time
import cv2

def record_video(duration=5):
    """Record a short video from camera."""
    print(colored(f"Recording {duration} seconds of video...", "cyan"))
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(colored("Camera not available!", "red"))
        return
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('recorded_video.avi', fourcc, 20.0, (640, 480))
    start_time = time.time()

    while int(time.time() - start_time) < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow("Recording...", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(colored("Video saved as recorded_video.avi", "green"))


def find_car_in_image(image_path):
    """Use Qwen-VL to describe image and detect if a car exists."""
    if not os.path.exists(image_path):
        print(colored("No image found to analyze.", "yellow"))
        return
    prompt = "Look carefully and tell me if there is a car in this image."
    describe_image_with_qwen(image_path, extra_prompt=prompt)


def detect_faces_and_bodies(image_path):
    """Detect human faces and full bodies in an image using OpenCV."""
    if not os.path.exists(image_path):
        print(colored("No image found to analyze.", "yellow"))
        return

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = body_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    for (x, y, w, h) in bodies:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imwrite("detected_faces_bodies.jpg", img)
    print(colored(f"Found {len(faces)} face(s) and {len(bodies)} body/bodies.", "green"))
    print(colored("Saved image as detected_faces_bodies.jpg", "cyan"))

# ===============================
# Vision Describe
# ===============================
def describe_image_with_qwen(image_path, extra_prompt=None):
    section("Vision Analysis", ">>")
    with open(image_path, "rb") as img_file:
        b64_image = base64.b64encode(img_file.read()).decode("utf-8")

    payload = {
        "model": vision_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": extra_prompt or "Describe this image in detail."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ],
            }
        ],
    }

    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        data = response.json()
    except Exception as e:
        print(colored(f"Failed to contact LM Studio: {e}", "red"))
        return

    if not response.ok:
        print(colored("LM Studio vision call failed:", "red"))
        print(json.dumps(data, indent=2))
        return

    try:
        description = data["choices"][0]["message"]["content"]
        print(colored("\nModel sees:\n", "cyan"), description)
        tts.tts_to_file(text=description, file_path="vision_output.wav", **getattr(tts, "extra_args", {}))
        play_audio("vision_output.wav")
    except Exception:
        print(colored("Unexpected LM Studio response format.", "yellow"))
        print(json.dumps(data, indent=2))

# ===============================
# Main Loop
# ===============================
camera_on_commands = ["turn camera on", "camera on"]
camera_off_commands = ["turn camera off", "camera off"]
take_picture_commands = ["take picture", "snap photo"]
take_screenshot_commands = ["take screenshot", "capture screen"]

record_video_commands = ["record video", "start recording", "record a short video"]
find_car_commands = ["find car", "detect car", "locate car in image", "is there a car"]
detect_faces_commands = ["detect faces", "find faces", "face detection", "detect people", "find body"]

describe_image_commands = ["describe image", "analyze image", "look at this image", "what's in the picture"]
describe_screenshot_commands = [
    "describe screenshot", "what you see on the screen", "what's on the screen",
    "describe screen shot", "analyze screenshot", "analyze screen shot"
]

while True:
    section("Recording", ">>")
    audio_data = record_audio(duration=5)
    sf.write("input.wav", audio_data, 16000)

    section("Transcribing", ">>")
    result = whisper_model.transcribe("input.wav")
    transcript = result.get("text", "").strip()
    print(colored("\n--- Transcript ---", "cyan"))
    print(transcript if transcript else "(no speech detected)")
    print(colored("-----------------", "cyan"))

    if not transcript:
        continue
    if STOP_PHRASE in transcript.lower():
        print(colored("Stop phrase detected. Exiting pipeline.", "red"))
        break

    command = transcript.lower()
    if any(cmd in command for cmd in camera_on_commands):
        turn_camera_on(); continue
    elif any(cmd in command for cmd in camera_off_commands):
        turn_camera_off(); continue
    elif any(cmd in command for cmd in take_picture_commands):
        take_picture(); continue
    elif any(cmd in command for cmd in take_screenshot_commands):
        take_screenshot(); continue
    elif any(cmd in command for cmd in describe_image_commands):
        if os.path.exists("captured.jpg"): describe_image_with_qwen("captured.jpg")
        else: print(colored("No camera image found.", "yellow"))
        continue
    elif any(cmd in command for cmd in describe_screenshot_commands):
        if os.path.exists("screenshot.jpg"): describe_image_with_qwen("screenshot.jpg")
        else: print(colored("No screenshot found.", "yellow"))
        continue

    elif any(cmd in command for cmd in record_video_commands):
        record_video(duration=5)
        continue

    elif any(cmd in command for cmd in find_car_commands):
        if os.path.exists("captured.jpg"):
            find_car_in_image("captured.jpg")
        elif os.path.exists("screenshot.jpg"):
            find_car_in_image("screenshot.jpg")
        else:
            print(colored("No image or screenshot found to search for a car.", "yellow"))
        continue

    elif any(cmd in command for cmd in detect_faces_commands):
        if os.path.exists("captured.jpg"):
            detect_faces_and_bodies("captured.jpg")
        elif os.path.exists("screenshot.jpg"):
            detect_faces_and_bodies("screenshot.jpg")
        else:
            print(colored("No image found for face/body detection.", "yellow"))
        continue

    elif "turn left" in command or "look left" in command:
        if arduino:
            arduino.write(b'left\n')
            print("Servo turning left")
            tts.tts_to_file(text="Turning left.", file_path="servo_feedback.wav", **getattr(tts, "extra_args", {}))
            play_audio("servo_feedback.wav")
        continue  # Skip LM Studio

    elif "turn right" in command or "look right" in command:
        if arduino:
            arduino.write(b'right\n')
            print("Servo turning right")
            tts.tts_to_file(text="Turning right.", file_path="servo_feedback.wav", **getattr(tts, "extra_args", {}))
            play_audio("servo_feedback.wav")
        continue  # Skip LM Studio

    elif "center" in command or "reset" in command:
        if arduino:
            arduino.write(b'center\n')
            print("Servo centered")
            tts.tts_to_file(text="Centering servo.", file_path="servo_feedback.wav", **getattr(tts, "extra_args", {}))
            play_audio("servo_feedback.wav")
        continue  # Skip LM Studio

    elif "look up" in command or "move up" in command or "up" in command:
        if arduino:
            arduino.write(b'up\n')
            print("Servo moving up")
            tts.tts_to_file(text="Looking up.", file_path="servo_feedback.wav", **getattr(tts, "extra_args", {}))
            play_audio("servo_feedback.wav")
        continue  # Skip LM Studio

    elif "look down" in command or "move down" in command or "down" in command:
        if arduino:
            arduino.write(b'down\n')
            print("Servo moving down")
            tts.tts_to_file(text="Looking down.", file_path="servo_feedback.wav", **getattr(tts, "extra_args", {}))
            play_audio("servo_feedback.wav")
        continue  # Skip LM Studio

    elif "level" in command:
        if arduino:
            arduino.write(b'level\n')
            print("Servo leveled")
            tts.tts_to_file(text="Leveling servo.", file_path="servo_feedback.wav", **getattr(tts, "extra_args", {}))
            play_audio("servo_feedback.wav")
        continue  # Skip LM Studio


    # ===============================
    # Default LM Studio Chat + Emotion
    # ===============================
    section("LM Studio", ">>")
    payload = {"model": text_model, "messages": [{"role": "user", "content": transcript}]}
    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        data = response.json()
        reply = data["choices"][0]["message"]["content"]
        print(colored("\n--- LM Studio Response ---", "cyan"))
        print(reply)
        print(colored("---------------------------", "cyan"))

        # Add Emotion Prefix
        emotion_prefixes = {
            "Happy": "with a cheerful and bright tone:",
            "Sad": "with a soft and sorrowful tone:",
            "Angry": "with a tense and forceful tone:",
            "Excited": "with an energetic and upbeat tone:",
            "Neutral": ""
        }

        emotion_prefix = ""
        if hasattr(tts, "extra_args") and "emotion" in tts.extra_args:
            selected_emotion = tts.extra_args["emotion"]
            emotion_prefix = emotion_prefixes.get(selected_emotion, "")

        speak_text = f"{emotion_prefix} {reply}" if emotion_prefix else reply

        # Speak with Emotion
        section("Generating Speech", ">>")
        tts.tts_to_file(text=speak_text, file_path="output.wav", **getattr(tts, "extra_args", {}))
        play_audio("output.wav")

    except Exception as e:
        print(colored(f"Error: {e}", "red"))
