from __future__ import annotations

import base64
import importlib
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv


def optional_import(module_name: str, attribute: str | None = None) -> Any:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    if attribute is None:
        return module
    return getattr(module, attribute, None)


cv2 = optional_import("cv2")
keyboard = optional_import("keyboard")
np = optional_import("numpy")
pyautogui = optional_import("pyautogui")
pyfiglet = optional_import("pyfiglet")
serial = optional_import("serial")
sd = optional_import("sounddevice")
sf = optional_import("soundfile")
torch = optional_import("torch")
whisper = optional_import("whisper")
TTS = optional_import("TTS.api", "TTS")
colored = optional_import("termcolor", "colored")


load_dotenv()


def banner(text: str, color: str = "cyan") -> None:
    ensure_dependency(pyfiglet, "pyfiglet")
    ensure_dependency(colored, "termcolor")
    print(colored(pyfiglet.figlet_format(text, font="slant"), color))


def section(title: str, prefix: str = ">>") -> None:
    ensure_dependency(colored, "termcolor")
    print(colored("\n" + "=" * 60, "yellow"))
    print(colored(f" {prefix}  {title}", "green", attrs=["bold"]))
    print(colored("=" * 60, "yellow"))


def ask_choice(prompt: str, options: dict[str, str]) -> str:
    ensure_dependency(colored, "termcolor")
    while True:
        choice = input(colored(f"{prompt} ({'/'.join(options)}): ", "yellow")).strip().upper()
        if choice in options:
            return choice
        print(colored("Invalid choice, try again.", "red"))


def env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}.") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


def env_float(name: str, default: float, minimum: float | None = None) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw_value!r}.") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


def env_str(name: str, default: str = "", *, required: bool = False) -> str:
    value = os.getenv(name, default).strip()
    if required and not value:
        raise ValueError(f"{name} is required.")
    return value


def validate_url(name: str, value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{name} must be a valid http(s) URL, got {value!r}.")
    return value.rstrip("/")


def ensure_dependency(module: Any, package_name: str) -> None:
    if module is None:
        raise RuntimeError(
            f"Missing dependency: {package_name}. Install requirements.txt before running the assistant."
        )


@dataclass(slots=True)
class AppConfig:
    lm_studio_base_url: str
    lm_studio_api_key: str
    lm_studio_timeout: float
    text_model: str
    vision_model: str
    whisper_model: str
    default_tts_model: str
    arduino_port: str
    arduino_baudrate: int
    arduino_timeout: float
    camera_index: int
    audio_sample_rate: int
    audio_duration: int
    audio_blocksize: int
    stop_phrase: str
    max_retries: int
    retry_delay: float
    output_dir: Path
    temp_dir: Path

    @classmethod
    def from_env(cls) -> "AppConfig":
        output_dir = Path(env_str("OUTPUT_DIR", "./output")).expanduser().resolve()
        temp_dir = Path(env_str("TEMP_DIR", "./temp")).expanduser().resolve()
        config = cls(
            lm_studio_base_url=validate_url(
                "LM_STUDIO_BASE_URL",
                env_str("LM_STUDIO_BASE_URL", "http://localhost:1234", required=True),
            ),
            lm_studio_api_key=env_str("LM_STUDIO_API_KEY", ""),
            lm_studio_timeout=env_float("LM_STUDIO_TIMEOUT", 60.0, minimum=1.0),
            text_model=env_str("TEXT_MODEL", "qwen2-vl-7b-instruct", required=True),
            vision_model=env_str("VISION_MODEL", "qwen2-vl-7b-instruct", required=True),
            whisper_model=env_str("WHISPER_MODEL", "small", required=True),
            default_tts_model=env_str(
                "DEFAULT_TTS_MODEL",
                "tts_models/en/ljspeech/tacotron2-DDC",
                required=True,
            ),
            arduino_port=env_str("ARDUINO_PORT", ""),
            arduino_baudrate=env_int("ARDUINO_BAUDRATE", 9600, minimum=1),
            arduino_timeout=env_float("ARDUINO_TIMEOUT", 1.0, minimum=0.1),
            camera_index=env_int("CAMERA_INDEX", 0, minimum=0),
            audio_sample_rate=env_int("AUDIO_SAMPLE_RATE", 16000, minimum=8000),
            audio_duration=env_int("AUDIO_DURATION", 5, minimum=1),
            audio_blocksize=env_int("AUDIO_BLOCKSIZE", 1024, minimum=256),
            stop_phrase=env_str("STOP_PHRASE", "stop listening", required=True).lower(),
            max_retries=env_int("MAX_RETRIES", 3, minimum=1),
            retry_delay=env_float("RETRY_DELAY", 1.0, minimum=0.0),
            output_dir=output_dir,
            temp_dir=temp_dir,
        )
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        return config


class VoiceAssistantApp:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        ensure_dependency(colored, "termcolor")
        ensure_dependency(torch, "torch")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if self.config.lm_studio_api_key:
            self.session.headers["Authorization"] = f"Bearer {self.config.lm_studio_api_key}"
        self.arduino: serial.Serial | None = None
        self.camera: cv2.VideoCapture | None = None
        self.preview_thread: threading.Thread | None = None
        self.preview_running = False
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.whisper_model: Any | None = None
        self.tts: TTS | None = None
        self.tts_extra_args: dict[str, Any] = {}
        self.command_handlers: list[tuple[tuple[str, ...], Any]] = [
            (("turn camera on", "camera on"), self.turn_camera_on),
            (("turn camera off", "camera off"), self.turn_camera_off),
            (("take picture", "snap photo"), self.take_picture),
            (("take screenshot", "capture screen"), self.take_screenshot),
            (("describe image", "analyze image", "look at this image", "what's in the picture"), self.describe_camera_image),
            (
                (
                    "describe screenshot",
                    "what you see on the screen",
                    "what's on the screen",
                    "describe screen shot",
                    "analyze screenshot",
                    "analyze screen shot",
                ),
                self.describe_saved_screenshot,
            ),
            (("record video", "start recording", "record a short video"), self.record_video),
            (("find car", "detect car", "locate car in image", "is there a car"), self.find_car),
            (("detect faces", "find faces", "face detection", "detect people", "find body"), self.detect_faces),
            (("turn left", "look left"), self.servo_left),
            (("turn right", "look right"), self.servo_right),
            (("center", "reset"), self.servo_center),
            (("look up", "move up"), self.servo_up),
            (("look down", "move down"), self.servo_down),
            (("level",), self.servo_level),
        ]

    def safe_output_path(self, filename: str) -> Path:
        return self.config.output_dir / filename

    def print_help(self) -> None:
        section("Voice Command Help")
        print(
            """Say any of the following:
Camera Control - "Turn camera on" / "Turn camera off" - "Take picture" / "Snap photo" - "Record a short video"
Screen Control - "Take screenshot" / "Capture screen"
Vision and Object Understanding - "Describe image" / "Analyze image" - "Describe screenshot" / "Analyze screenshot"
Object & Person Detection - "Find car in image" - "Detect faces" / "Detect body"
Control - "Stop listening" exits the program
"""
        )

    def initialise_hardware(self) -> None:
        if not self.config.arduino_port:
            print(colored("Arduino disabled: ARDUINO_PORT not set.", "yellow"))
            return
        ensure_dependency(serial, "pyserial")
        try:
            self.arduino = serial.Serial(
                self.config.arduino_port,
                self.config.arduino_baudrate,
                timeout=self.config.arduino_timeout,
            )
            time.sleep(2)
            print(colored(f"Connected to Arduino on {self.config.arduino_port}", "green"))
        except serial.SerialException as exc:
            print(colored(f"Arduino connection failed: {exc}", "yellow"))
            self.arduino = None

    def initialise_models(self) -> None:
        ensure_dependency(whisper, "openai-whisper")
        ensure_dependency(TTS, "TTS")
        section("Loading Whisper model")
        self.whisper_model = whisper.load_model(self.config.whisper_model)
        print(colored("Whisper ready.", "cyan", attrs=["bold"]))

        banner("COQUI PIPELINE", "magenta")
        print(colored(f"Using device: {self.device}", "magenta", attrs=["bold"]))
        print(colored("Would you like to:", "cyan"))
        print(colored("1. Auto-load instantly (default voice, no setup)", "yellow"))
        print(colored("2. Try advanced setup (custom voice, cloning, emotions, language)", "yellow"))
        mode_choice = input(colored("Enter 1 or 2: ", "cyan")).strip()
        auto_mode = mode_choice == "1"

        if auto_mode:
            section("Auto Mode")
            selected_voice = self.config.default_tts_model
            self.tts = TTS(selected_voice, progress_bar=False, gpu=(self.device == "cuda"))
            self.tts_extra_args = {}
            print(colored("Auto-load complete (default Coqui voice active).", "green"))
            return

        banner("VOICE SETUP", "cyan")
        voice_choice = ask_choice("Keep the default voice?", {"Y": "Yes", "N": "No"})
        if voice_choice == "N":
            available_voices = [
                self.config.default_tts_model,
                "tts_models/en/ljspeech/glow-tts",
                "tts_models/en/vctk/vits",
                "tts_models/multilingual/multi-dataset/your_tts",
            ]
            for idx, voice_name in enumerate(available_voices, start=1):
                print(colored(f"  {idx}. {voice_name}", "cyan"))
            selected_voice = self.choose_indexed_value("Select a voice number", available_voices)
        else:
            selected_voice = self.config.default_tts_model

        section("Loading Coqui TTS")
        self.tts = TTS(selected_voice, progress_bar=False, gpu=(self.device == "cuda"))
        print(colored("Coqui TTS ready.", "cyan", attrs=["bold"]))

        selected_speaker: str | None = None
        selected_language: str | None = None
        selected_emotion: str | None = None
        speaker_wav: str | None = None

        if getattr(self.tts, "speakers", None):
            section("Multi-Speaker Model Detected")
            selected_speaker = self.choose_indexed_value("Select speaker number", list(self.tts.speakers))
            print(colored(f"Selected speaker: {selected_speaker}", "green"))

        if getattr(self.tts, "languages", None):
            section("Multi-Lingual Model Detected")
            selected_language = self.choose_indexed_value("Select language number", list(self.tts.languages))
            print(colored(f"Selected language: {selected_language}", "green"))

        emotions = ["Neutral", "Happy", "Sad", "Angry", "Excited"]
        section("Voice Emotions")
        if ask_choice("Would you like to add an emotion?", {"Y": "Yes", "N": "No"}) == "Y":
            selected_emotion = self.choose_indexed_value("Select emotion number", emotions)
            print(colored(f"Selected emotion: {selected_emotion}", "green"))

        section("Voice Cloning")
        if ask_choice("Would you like to clone a voice?", {"Y": "Yes", "N": "No"}) == "Y":
            candidate = Path(input(colored("Enter path to your 6-sec .wav file: ", "yellow")).strip()).expanduser()
            if candidate.is_file() and candidate.suffix.lower() == ".wav":
                speaker_wav = str(candidate.resolve())
            else:
                print(colored("Voice cloning skipped: invalid .wav path.", "yellow"))

        self.tts_extra_args = {}
        if selected_emotion:
            self.tts_extra_args["emotion"] = selected_emotion
        if selected_speaker:
            self.tts_extra_args["speaker"] = selected_speaker
        if selected_language:
            self.tts_extra_args["language"] = selected_language
        if speaker_wav:
            self.tts_extra_args["speaker_wav"] = speaker_wav

        print(colored("Voice setup complete.", "green", attrs=["bold"]))

    def choose_indexed_value(self, prompt: str, options: list[str]) -> str:
        for index, option in enumerate(options, start=1):
            print(colored(f" {index}. {option}", "green"))
        while True:
            try:
                selected = int(input(colored(f"\n{prompt}: ", "yellow")).strip())
            except ValueError:
                selected = -1
            if 1 <= selected <= len(options):
                return options[selected - 1]
            print(colored("Invalid selection, try again.", "red"))

    def record_audio(self) -> np.ndarray:
        ensure_dependency(sd, "sounddevice")
        ensure_dependency(np, "numpy")
        print(colored("Mic active... speak now!", "cyan", attrs=["bold"]))
        audio = sd.rec(
            int(self.config.audio_duration * self.config.audio_sample_rate),
            samplerate=self.config.audio_sample_rate,
            channels=1,
            dtype="float32",
            blocking=True,
        )
        return np.squeeze(audio)

    def play_audio(self, file_path: Path) -> None:
        ensure_dependency(sd, "sounddevice")
        ensure_dependency(sf, "soundfile")
        ensure_dependency(keyboard, "keyboard")
        data, sample_rate = sf.read(file_path, dtype="float32")
        channels = data.shape[1] if data.ndim > 1 else 1
        self.pause_event.clear()
        self.stop_event.clear()

        def audio_worker() -> None:
            with sd.OutputStream(
                samplerate=sample_rate,
                channels=channels,
                blocksize=self.config.audio_blocksize,
            ) as stream:
                index = 0
                while index < len(data):
                    if self.stop_event.is_set():
                        return
                    if self.pause_event.is_set():
                        time.sleep(0.1)
                        continue
                    end_index = min(index + self.config.audio_blocksize, len(data))
                    stream.write(data[index:end_index])
                    index = end_index

        worker = threading.Thread(target=audio_worker, daemon=True)
        worker.start()
        print(colored("Playing... Press 'p' to pause/resume, 's' to stop.", "cyan"))
        while worker.is_alive():
            try:
                if keyboard.is_pressed("p"):
                    if self.pause_event.is_set():
                        self.pause_event.clear()
                        print(colored("Resumed", "green"))
                    else:
                        self.pause_event.set()
                        print(colored("Paused", "yellow"))
                    time.sleep(0.3)
                elif keyboard.is_pressed("s"):
                    self.stop_event.set()
                    print(colored("Stopped", "yellow"))
                    break
            except RuntimeError:
                self.stop_event.set()
                print(colored("Keyboard hooks unavailable; stopping audio playback.", "yellow"))
                break
            time.sleep(0.1)
        worker.join()

    def camera_preview(self) -> None:
        ensure_dependency(cv2, "opencv-python")
        while self.preview_running and self.camera is not None:
            success, frame = self.camera.read()
            if not success:
                print(colored("Camera preview ended: failed to read frame.", "yellow"))
                break
            cv2.imshow("Camera Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.preview_running = False
        cv2.destroyAllWindows()

    def turn_camera_on(self) -> bool:
        ensure_dependency(cv2, "opencv-python")
        if self.camera is not None:
            print(colored("Camera already ON.", "yellow"))
            return True
        camera = cv2.VideoCapture(self.config.camera_index)
        if not camera.isOpened():
            print(colored("Failed to open camera.", "red"))
            camera.release()
            return False
        self.camera = camera
        self.preview_running = True
        self.preview_thread = threading.Thread(target=self.camera_preview, daemon=True)
        self.preview_thread.start()
        print(colored("Camera turned ON (preview active).", "green"))
        return True

    def turn_camera_off(self) -> None:
        if self.camera is None:
            print(colored("Camera already OFF.", "yellow"))
            return
        ensure_dependency(cv2, "opencv-python")
        self.preview_running = False
        time.sleep(0.2)
        self.camera.release()
        self.camera = None
        cv2.destroyAllWindows()
        print(colored("Camera turned OFF.", "yellow"))

    def take_picture(self) -> Path | None:
        ensure_dependency(cv2, "opencv-python")
        if self.camera is None:
            print(colored("Camera is OFF. Turn it ON first.", "yellow"))
            return None
        success, frame = self.camera.read()
        if not success:
            print(colored("Failed to capture image.", "red"))
            return None
        image_path = self.safe_output_path("captured.jpg")
        cv2.imwrite(str(image_path), frame)
        print(colored(f"Image saved as {image_path}", "green"))
        return image_path

    def take_screenshot(self) -> Path | None:
        ensure_dependency(pyautogui, "pyautogui")
        image_path = self.safe_output_path("screenshot.jpg")
        try:
            screenshot = pyautogui.screenshot()
            screenshot.save(image_path)
        except OSError as exc:
            print(colored(f"Failed to capture screenshot: {exc}", "red"))
            return None
        print(colored(f"Screenshot saved as {image_path}", "green"))
        return image_path

    def record_video(self, duration: int | None = None) -> Path | None:
        ensure_dependency(cv2, "opencv-python")
        actual_duration = duration or self.config.audio_duration
        print(colored(f"Recording {actual_duration} seconds of video...", "cyan"))
        capture = cv2.VideoCapture(self.config.camera_index)
        if not capture.isOpened():
            print(colored("Camera not available.", "red"))
            capture.release()
            return None

        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        output_path = self.safe_output_path("recorded_video.avi")
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"XVID"),
            20.0,
            (frame_width, frame_height),
        )
        start_time = time.time()

        try:
            while time.time() - start_time < actual_duration:
                success, frame = capture.read()
                if not success:
                    print(colored("Video recording interrupted: failed to read frame.", "yellow"))
                    break
                writer.write(frame)
                cv2.imshow("Recording...", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            capture.release()
            writer.release()
            cv2.destroyAllWindows()

        print(colored(f"Video saved as {output_path}", "green"))
        return output_path

    def detect_faces_and_bodies(self, image_path: Path) -> Path | None:
        ensure_dependency(cv2, "opencv-python")
        if not image_path.is_file():
            print(colored("No image found to analyze.", "yellow"))
            return None

        image = cv2.imread(str(image_path))
        if image is None:
            print(colored("Failed to load image for detection.", "red"))
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
        if face_cascade.empty() or body_cascade.empty():
            print(colored("OpenCV cascades are unavailable on this system.", "red"))
            return None

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for x, y, width, height in faces:
            cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)
        for x, y, width, height in bodies:
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

        output_path = self.safe_output_path("detected_faces_bodies.jpg")
        cv2.imwrite(str(output_path), image)
        print(colored(f"Found {len(faces)} face(s) and {len(bodies)} body/bodies.", "green"))
        print(colored(f"Saved image as {output_path}", "cyan"))
        return output_path

    def encode_image(self, image_path: Path) -> str | None:
        if not image_path.is_file():
            print(colored("No image found to analyze.", "yellow"))
            return None
        with image_path.open("rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def lm_studio_post(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any] | None:
        endpoint = f"{self.config.lm_studio_base_url}/v1/chat/completions"
        request_timeout = timeout or self.config.lm_studio_timeout
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.session.post(endpoint, json=payload, timeout=request_timeout)
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt == self.config.max_retries:
                    break
                time.sleep(self.config.retry_delay)

        print(colored(f"LM Studio request failed after {self.config.max_retries} attempt(s): {last_error}", "red"))
        return None

    def describe_image_with_qwen(self, image_path: Path, extra_prompt: str | None = None) -> str | None:
        section("Vision Analysis")
        encoded_image = self.encode_image(image_path)
        if encoded_image is None:
            return None

        payload = {
            "model": self.config.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": extra_prompt or "Describe this image in detail."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                        },
                    ],
                }
            ],
        }

        data = self.lm_studio_post(payload)
        if data is None:
            return None

        try:
            description = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            print(colored("Unexpected LM Studio response format.", "yellow"))
            return None

        print(colored("\nModel sees:\n", "cyan"), description)
        self.speak_text(description, "vision_output.wav")
        return description

    def transcribe_audio(self, audio_path: Path) -> str:
        if self.whisper_model is None:
            raise RuntimeError("Whisper model is not initialised.")
        result = self.whisper_model.transcribe(str(audio_path))
        return str(result.get("text", "")).strip()

    def speak_text(self, text: str, output_name: str) -> None:
        ensure_dependency(TTS, "TTS")
        if self.tts is None:
            raise RuntimeError("TTS engine is not initialised.")
        output_path = self.safe_output_path(output_name)
        self.tts.tts_to_file(text=text, file_path=str(output_path), **self.tts_extra_args)
        self.play_audio(output_path)

    def describe_camera_image(self) -> None:
        captured_image = self.safe_output_path("captured.jpg")
        if captured_image.is_file():
            self.describe_image_with_qwen(captured_image)
            return
        print(colored("No camera image found.", "yellow"))

    def describe_saved_screenshot(self) -> None:
        screenshot = self.safe_output_path("screenshot.jpg")
        if screenshot.is_file():
            self.describe_image_with_qwen(screenshot)
            return
        print(colored("No screenshot found.", "yellow"))

    def find_car(self) -> None:
        prompt = "Look carefully and tell me if there is a car in this image."
        for candidate in (self.safe_output_path("captured.jpg"), self.safe_output_path("screenshot.jpg")):
            if candidate.is_file():
                self.describe_image_with_qwen(candidate, extra_prompt=prompt)
                return
        print(colored("No image or screenshot found to search for a car.", "yellow"))

    def detect_faces(self) -> None:
        for candidate in (self.safe_output_path("captured.jpg"), self.safe_output_path("screenshot.jpg")):
            if candidate.is_file():
                self.detect_faces_and_bodies(candidate)
                return
        print(colored("No image found for face/body detection.", "yellow"))

    def send_servo_command(self, command: bytes, spoken_text: str, console_text: str) -> None:
        ensure_dependency(serial, "pyserial")
        if self.arduino is None:
            print(colored("Servo command ignored: Arduino is not connected.", "yellow"))
            return
        try:
            self.arduino.write(command)
        except serial.SerialException as exc:
            print(colored(f"Servo command failed: {exc}", "red"))
            return
        print(colored(console_text, "green"))
        self.speak_text(spoken_text, "servo_feedback.wav")

    def servo_left(self) -> None:
        self.send_servo_command(b"left\n", "Turning left.", "Servo turning left")

    def servo_right(self) -> None:
        self.send_servo_command(b"right\n", "Turning right.", "Servo turning right")

    def servo_center(self) -> None:
        self.send_servo_command(b"center\n", "Centering servo.", "Servo centered")

    def servo_up(self) -> None:
        self.send_servo_command(b"up\n", "Looking up.", "Servo moving up")

    def servo_down(self) -> None:
        self.send_servo_command(b"down\n", "Looking down.", "Servo moving down")

    def servo_level(self) -> None:
        self.send_servo_command(b"level\n", "Leveling servo.", "Servo leveled")

    def route_command(self, transcript: str) -> bool:
        command = transcript.lower()
        for phrases, handler in self.command_handlers:
            if any(phrase in command for phrase in phrases):
                handler()
                return True
        return False

    def run_chat(self, transcript: str) -> None:
        section("LM Studio")
        payload = {
            "model": self.config.text_model,
            "messages": [{"role": "user", "content": transcript}],
        }
        data = self.lm_studio_post(payload, timeout=min(self.config.lm_studio_timeout, 30.0))
        if data is None:
            return

        try:
            reply = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            print(colored("Unexpected LM Studio response format.", "yellow"))
            return

        print(colored("\n--- LM Studio Response ---", "cyan"))
        print(reply)
        print(colored("---------------------------", "cyan"))

        emotion_prefixes = {
            "Happy": "with a cheerful and bright tone:",
            "Sad": "with a soft and sorrowful tone:",
            "Angry": "with a tense and forceful tone:",
            "Excited": "with an energetic and upbeat tone:",
            "Neutral": "",
        }
        selected_emotion = str(self.tts_extra_args.get("emotion", ""))
        emotion_prefix = emotion_prefixes.get(selected_emotion, "")
        speak_text = f"{emotion_prefix} {reply}".strip() if emotion_prefix else reply

        section("Generating Speech")
        self.speak_text(speak_text, "output.wav")

    def cleanup(self) -> None:
        if self.camera is not None:
            self.turn_camera_off()
        if self.arduino is not None:
            self.arduino.close()
            self.arduino = None
        self.session.close()

    def run(self) -> None:
        banner("COQUI PIPELINE", "magenta")
        self.print_help()
        self.initialise_hardware()
        self.initialise_models()
        ensure_dependency(sf, "soundfile")

        input_audio_path = self.safe_output_path("input.wav")
        try:
            while True:
                section("Recording")
                audio_data = self.record_audio()
                sf.write(str(input_audio_path), audio_data, self.config.audio_sample_rate)

                section("Transcribing")
                transcript = self.transcribe_audio(input_audio_path)
                print(colored("\n--- Transcript ---", "cyan"))
                print(transcript if transcript else "(no speech detected)")
                print(colored("-----------------", "cyan"))

                if not transcript:
                    continue
                if self.config.stop_phrase in transcript.lower():
                    print(colored("Stop phrase detected. Exiting pipeline.", "red"))
                    break
                if self.route_command(transcript):
                    continue
                self.run_chat(transcript)
        finally:
            self.cleanup()


def main() -> None:
    config = AppConfig.from_env()
    app = VoiceAssistantApp(config)
    app.run()


if __name__ == "__main__":
    main()
