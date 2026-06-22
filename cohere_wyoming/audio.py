"""Audio helpers shared by the HTTP and Wyoming entrypoints."""

from __future__ import annotations

import io
import os
import subprocess
import tempfile

import librosa
import numpy as np
import soundfile as sf


TARGET_SAMPLE_RATE = 16000
SILENCE_RMS_THRESHOLD = 0.002
SILENCE_PEAK_THRESHOLD = 0.01
SILENCE_FRAME_MS = 30
SILENCE_HOP_MS = 10
MIN_ACTIVE_SPEECH_SECONDS = 0.08
MIN_ACTIVE_STREAK_SECONDS = 0.05


def normalize_audio(audio_data: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Convert audio to mono float32 at 16 kHz."""
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    audio_data = np.asarray(audio_data, dtype=np.float32)
    if sample_rate != TARGET_SAMPLE_RATE:
        audio_data = librosa.resample(
            audio_data, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE
        )
        sample_rate = TARGET_SAMPLE_RATE

    return audio_data, sample_rate


def read_audio_to_numpy(file_bytes: bytes, filename: str = "audio") -> tuple[np.ndarray, int]:
    """Decode common audio container formats into mono 16 kHz float32 samples."""
    try:
        audio_io = io.BytesIO(file_bytes)
        audio_data, sample_rate = sf.read(audio_io, dtype="float32")
        return normalize_audio(audio_data, sample_rate)
    except Exception:
        pass

    librosa_error: Exception | None = None
    try:
        audio_io = io.BytesIO(file_bytes)
        audio_data, sample_rate = librosa.load(
            audio_io, sr=TARGET_SAMPLE_RATE, mono=True
        )
        return np.asarray(audio_data, dtype=np.float32), sample_rate
    except Exception as error:
        librosa_error = error

    # Decode via a temp file so ffmpeg can seek (required for m4a/mp4 moov atoms,
    # which fail when piped through stdin).
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1] or ".bin", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        process = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                tmp_path,
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ac",
                "1",
                "-ar",
                str(TARGET_SAMPLE_RATE),
                "pipe:1",
            ],
            capture_output=True,
            check=True,
        )
        audio_data = np.frombuffer(process.stdout, dtype=np.float32)
        if audio_data.size == 0:
            raise ValueError("ffmpeg produced empty audio output")
        return audio_data, TARGET_SAMPLE_RATE
    except Exception as ffmpeg_error:
        raise ValueError(
            f"Could not read audio file '{filename}'. "
            "Supported formats: WAV, MP3, FLAC, OGG, WebM, M4A and other ffmpeg-decodable "
            f"audio. soundfile/librosa error: {librosa_error}. ffmpeg error: {ffmpeg_error}"
        ) from ffmpeg_error
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def pcm16le_to_float32(
    audio_bytes: bytes,
    *,
    sample_rate: int,
    channels: int,
    width: int,
) -> tuple[np.ndarray, int]:
    """Convert raw PCM16LE bytes from Wyoming to mono 16 kHz float32 samples."""
    if width != 2:
        raise ValueError(f"Unsupported PCM width: {width}. Only 16-bit PCM is supported.")

    if channels <= 0:
        raise ValueError(f"Unsupported channel count: {channels}")

    audio = np.frombuffer(audio_bytes, dtype="<i2")
    if channels > 1:
        remainder = len(audio) % channels
        if remainder:
            audio = audio[:-remainder]
        audio = audio.reshape(-1, channels).mean(axis=1)
    else:
        audio = audio.astype(np.float32)

    audio = np.asarray(audio, dtype=np.float32) / 32768.0
    return normalize_audio(audio, sample_rate)


def is_effectively_silent(audio_data: np.ndarray) -> bool:
    """Return True when the decoded audio is effectively silence."""
    if audio_data.size == 0:
        return True

    audio_data = np.asarray(audio_data, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(audio_data, dtype=np.float32))))
    peak = float(np.max(np.abs(audio_data)))
    if rms < SILENCE_RMS_THRESHOLD and peak < SILENCE_PEAK_THRESHOLD:
        return True

    frame_length = max(1, int(TARGET_SAMPLE_RATE * SILENCE_FRAME_MS / 1000))
    hop_length = max(1, int(TARGET_SAMPLE_RATE * SILENCE_HOP_MS / 1000))
    if audio_data.size < frame_length:
        return rms < (SILENCE_RMS_THRESHOLD * 2.0) and peak < (SILENCE_PEAK_THRESHOLD * 1.5)

    frame_rms = []
    for start in range(0, audio_data.size - frame_length + 1, hop_length):
        frame = audio_data[start : start + frame_length]
        frame_rms.append(float(np.sqrt(np.mean(np.square(frame, dtype=np.float32)))))

    if not frame_rms:
        return True

    frame_rms_array = np.asarray(frame_rms, dtype=np.float32)
    noise_floor = float(np.percentile(frame_rms_array, 20))
    absolute_threshold = SILENCE_RMS_THRESHOLD * 2.5
    relative_cap = max(absolute_threshold, rms * 0.6)
    active_threshold = min(max(absolute_threshold, noise_floor * 3.0), relative_cap)
    active_frames = frame_rms_array >= active_threshold
    active_duration = float(np.count_nonzero(active_frames) * hop_length / TARGET_SAMPLE_RATE)

    max_streak = 0
    current_streak = 0
    for is_active in active_frames:
        if is_active:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    active_streak_duration = float(max_streak * hop_length / TARGET_SAMPLE_RATE)
    return (
        active_duration < MIN_ACTIVE_SPEECH_SECONDS
        or active_streak_duration < MIN_ACTIVE_STREAK_SECONDS
    )
