"""The stone-on-board click, played from a bundled recording.

``katago/assets/stone_click.wav`` holds the recorded click (mono 16-bit PCM).
If that file is missing, a wooden "tock" is synthesised with the standard
library instead and cached under the temp directory, so the effect never
disappears.  Windows uses ``winsound``, macOS shells out to the bundled
``afplay``; anything else falls back to the Tk bell the caller passes in.
Playback never blocks the UI and never raises -- going silent is always an
acceptable outcome.
"""

from __future__ import annotations

import array
import io
import math
import os
import random
import subprocess
import sys
import tempfile
import threading
import wave

CLICK_SAMPLE_RATE = 22050
CLICK_SECONDS = 0.26
#: the length is part of the name so a new render never fights an older file
CLICK_FILE = f"katago_stone_click_{int(CLICK_SECONDS * 1000)}ms.wav"

#: the bundled recording, resolved next to the package so it also works from source
ASSET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    "stone_click.wav",
)

_lock = threading.Lock()
_path: str | None = None


def render_click(
    seconds: float = CLICK_SECONDS, sample_rate: int = CLICK_SAMPLE_RATE
) -> bytes:
    """Build a mono 16-bit wav: a stone landing on a wooden board.

    A real click is a low "tock" with a dull contact noise on top, both gliding
    down a little as the contact settles.  A short high tone reads as a beep,
    so the body sits around 230 Hz and rings for ~90 ms.
    """
    count = int(sample_rate * seconds)
    rng = random.Random(20240607)  # the same click on every run
    samples = array.array("h", bytes(2 * count))
    step = 1.0 / sample_rate
    # one-pole low pass on the noise: ~770 Hz cutoff, a dull wooden knock
    # instead of the bright "tss" of raw white noise
    dull = 0.0
    phase1 = phase2 = phase3 = 0.0
    for i in range(count):
        t = i / sample_rate
        strike = rng.random() * 2.0 - 1.0
        dull += 0.18 * (strike - dull)
        # a dull thud plus a very short bright edge: the edge is what makes the
        # click readable, the thud is what makes it wooden
        contact = (dull * 2.6 + strike * 0.45) * math.exp(-t / 0.006)
        # board modes at ~230 / 465 / 900 Hz, each decaying a little faster
        glide = 1.0 + 0.16 * math.exp(-t / 0.045)
        phase1 += 2.0 * math.pi * 232.0 * glide * step
        phase2 += 2.0 * math.pi * 466.0 * glide * step
        phase3 += 2.0 * math.pi * 902.0 * glide * step
        body = (
            0.62 * math.sin(phase1) * math.exp(-t / 0.090)
            + 0.28 * math.sin(phase2 + 0.5) * math.exp(-t / 0.055)
            + 0.12 * math.sin(phase3 + 1.2) * math.exp(-t / 0.028)
        )
        value = math.tanh(1.4 * (body + contact)) * 0.82
        # fade the tail so the file never ends on a step (which would click)
        value *= min(1.0, (seconds - t) / 0.030)
        samples[i] = int(max(-1.0, min(1.0, value)) * 32767)
    if sys.byteorder == "big":
        samples.byteswap()  # wav samples are little-endian
    buf = io.BytesIO()
    with wave.open(buf, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(samples.tobytes())
    return buf.getvalue()


def prepare_click() -> str:
    """Return the wav to play: the bundled recording, else a rendered fallback."""
    global _path
    if os.path.exists(ASSET_PATH):
        return ASSET_PATH
    if _path is not None and os.path.exists(_path):
        return _path
    with _lock:
        if _path is not None and os.path.exists(_path):
            return _path
        target = os.path.join(tempfile.gettempdir(), CLICK_FILE)
        scratch = f"{target}.{os.getpid()}.tmp"
        with open(scratch, "wb") as fh:
            fh.write(render_click())
        # atomic, so a second process never plays a half-written file
        os.replace(scratch, target)
        _path = target
    return _path


def play_stone_click(bell=None) -> bool:
    """Play the click without blocking; False means nothing was played.

    There is deliberately no "gave up" latch: a single failure (an antivirus
    holding the wav, a device that appeared later) must not mute the rest of
    the session.
    """
    try:
        if sys.platform == "win32":
            import winsound

            try:
                winsound.PlaySound(
                    prepare_click(),
                    winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
                )
            except RuntimeError:
                # PlaySound reports a device / file problem this way; a system
                # beep is a poor click but better at proving audio is alive
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
                return False
        elif sys.platform == "darwin":
            subprocess.Popen(
                ["afplay", prepare_click()],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif bell is not None:
            bell()
        return True
    except Exception:  # noqa: BLE001 - the GUI keeps working without sound
        return False
