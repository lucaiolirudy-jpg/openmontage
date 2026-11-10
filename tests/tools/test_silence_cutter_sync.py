"""silence_cutter remove mode keeps audio and video in sync across many cuts."""
import json
import shutil
import subprocess

import pytest

from tools.video.silence_cutter import SilenceCutter

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def _duration(path, stream):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries",
         "stream=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def test_jump_cut_keeps_streams_aligned(tmp_path):
    src = tmp_path / "src.mp4"
    # 12 x (0.7s tone + 0.6s silence) at 25 fps: many cuts on a non-30fps source.
    tone = "sine=f=440:d=0.7,apad=pad_dur=0.6"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=s=160x120:r=25:d=15.6",
         "-f", "lavfi", "-i", f"{tone},aloop=loop=11:size=62400",
         "-t", "15.6", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         str(src)],
        check=True,
    )
    out = tmp_path / "cut.mp4"
    result = SilenceCutter().execute({
        "input_path": str(src), "output_path": str(out), "mode": "remove",
        "silence_threshold_db": -40, "min_silence_duration": 0.3, "padding_seconds": 0.05,
    })
    assert result.success, result.error
    assert result.data["speech_segments"] >= 10

    video, audio = _duration(out, "v:0"), _duration(out, "a:0")
    assert abs(video - audio) < 0.06  # within ~1 frame + AAC padding

    seg_map = json.loads((tmp_path / "cut.mp4.segments.json").read_text())
    assert abs(seg_map["output_duration"] - video) < 0.05
    starts = [s["output_start"] for s in seg_map["segments"]]
    assert starts == sorted(starts)
