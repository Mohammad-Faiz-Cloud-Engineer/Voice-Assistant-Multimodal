# Voice Assistant Multimodal

A multimodal AI voice assistant with speech recognition, text-to-speech, computer vision, and hardware control capabilities.

**Creator/Author:** [Mohammad Faiz](https://github.com/Mohammad-Faiz-Cloud-Engineer)

**Repository:** [https://github.com/Mohammad-Faiz-Cloud-Engineer/Voice-Assistant-Multimodal](https://github.com/Mohammad-Faiz-Cloud-Engineer/Voice-Assistant-Multimodal)

## Features

- **Speech Recognition**: Whisper-based voice command recognition
- **Text-to-Speech**: Coqui TTS with emotion support and voice cloning
- **Computer Vision**: Image analysis, object detection, face detection
- **Camera Control**: Real-time camera preview and image capture
- **Screen Capture**: Screenshot functionality
- **Video Recording**: Short video recording capability
- **Hardware Control**: Arduino servo motor control
- **LM Studio Integration**: Local LLM for conversational AI

## Requirements

- Python 3.14+
- CUDA-capable GPU (optional, for faster inference)
- Webcam (for camera features)
- Arduino (optional, for servo control)
- LM Studio running locally on port 1234

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and configure:
   ```bash
   cp .env.example .env
   ```

5. Edit `.env` with your configuration
   - Set `ARDUINO_PORT` only if you want servo control enabled
   - Keep `LM_STUDIO_BASE_URL` on a trusted local or private endpoint
   - Adjust `OUTPUT_DIR` and `TEMP_DIR` if you need non-default storage paths

## Usage

```bash
python3 voice_assistant_multimodal.py
```

### Voice Commands

- **Camera**: "Turn camera on/off", "Take picture", "Record video"
- **Screen**: "Take screenshot", "Capture screen"
- **Vision**: "Describe image", "Analyze screenshot"
- **Detection**: "Find car", "Detect faces"
- **Servo**: "Turn left/right", "Look up/down", "Center"
- **Exit**: "Stop listening"

## Configuration

All configuration is managed through environment variables in `.env`:

- `LM_STUDIO_BASE_URL`: LM Studio API endpoint
- `ARDUINO_PORT`: Optional serial port for Arduino (e.g., COM3, /dev/ttyUSB0)
- `CAMERA_INDEX`: Camera device index (usually 0)
- `WHISPER_MODEL`: Whisper model size (tiny, base, small, medium, large)
- `OUTPUT_DIR` / `TEMP_DIR`: Writable directories for generated media files

## Security Notes

- Never commit `.env` file to version control
- Use HTTPS for production API endpoints
- Keep LM Studio bound to trusted interfaces only
- Run with minimal required permissions

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Please ensure code passes all security and quality checks.

---

Made by Mohammad Faiz
