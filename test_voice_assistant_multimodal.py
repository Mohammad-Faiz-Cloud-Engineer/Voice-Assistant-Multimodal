import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from voice_assistant_multimodal import AppConfig, VoiceAssistantApp


class AppConfigTests(unittest.TestCase):
    def test_from_env_creates_directories_and_normalises_url(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            scratch_dir = Path(temp_dir) / "temp"
            env = {
                "LM_STUDIO_BASE_URL": "http://localhost:1234/",
                "TEXT_MODEL": "text-model",
                "VISION_MODEL": "vision-model",
                "WHISPER_MODEL": "small",
                "DEFAULT_TTS_MODEL": "tts_models/en/ljspeech/tacotron2-DDC",
                "STOP_PHRASE": "stop listening",
                "OUTPUT_DIR": str(output_dir),
                "TEMP_DIR": str(scratch_dir),
            }
            with patch.dict(os.environ, env, clear=False):
                config = AppConfig.from_env()
                self.assertEqual(config.lm_studio_base_url, "http://localhost:1234")
                self.assertTrue(config.output_dir.is_dir())
                self.assertTrue(config.temp_dir.is_dir())

    def test_from_env_rejects_invalid_base_url(self) -> None:
        env = {
            "LM_STUDIO_BASE_URL": "localhost:1234",
            "TEXT_MODEL": "text-model",
            "VISION_MODEL": "vision-model",
            "WHISPER_MODEL": "small",
            "DEFAULT_TTS_MODEL": "tts_models/en/ljspeech/tacotron2-DDC",
            "STOP_PHRASE": "stop listening",
        }
        with patch.dict(os.environ, env, clear=False):
            with self.assertRaises(ValueError):
                AppConfig.from_env()


class RouteCommandTests(unittest.TestCase):
    def make_app(self) -> VoiceAssistantApp:
        base = Path(tempfile.mkdtemp())
        env = {
            "LM_STUDIO_BASE_URL": "http://localhost:1234",
            "TEXT_MODEL": "text-model",
            "VISION_MODEL": "vision-model",
            "WHISPER_MODEL": "small",
            "DEFAULT_TTS_MODEL": "tts_models/en/ljspeech/tacotron2-DDC",
            "STOP_PHRASE": "stop listening",
            "OUTPUT_DIR": str(base / "output"),
            "TEMP_DIR": str(base / "temp"),
        }
        with patch.dict(os.environ, env, clear=False):
            config = AppConfig.from_env()

        with patch("voice_assistant_multimodal.ensure_dependency"), patch(
            "voice_assistant_multimodal.torch"
        ) as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            return VoiceAssistantApp(config)

    def test_route_command_matches_servo_phrase_without_generic_false_positive(self) -> None:
        app = self.make_app()
        called = {"up": 0}

        def mark_up() -> None:
            called["up"] += 1

        app.command_handlers = [(("look up", "move up"), mark_up)]

        self.assertFalse(app.route_command("what is the next backup status"))
        self.assertEqual(called["up"], 0)
        self.assertTrue(app.route_command("please look up now"))
        self.assertEqual(called["up"], 1)


if __name__ == "__main__":
    unittest.main()
