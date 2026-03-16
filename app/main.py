import os
import sys
import base64
import shutil
import subprocess
import tempfile
import logging

from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)


app = FastAPI(title="ARMS API")

WAN_PATH = "/opt/Wan2.2"
CKPT_DIR = "/opt/models/Wan2.2-S2V-14B"
DEFAULT_ANCHOR = "/app/resources/img.png"
DEFAULT_SIZE = "1024*704"
INFERENCE_TIMEOUT = 36000


class GenerateVideoRequest(BaseModel):
    audioData: str
    size: Optional[str] = None


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/generate-video")
def generate_video(request: GenerateVideoRequest):

    logger.info("video generate start.")
    tmp_dir = tempfile.mkdtemp(prefix="wan_video_")

    audio_bytes = base64.b64decode(request.audioData)
    wav_path = os.path.join(tmp_dir, "input.wav")
    with open(wav_path, "wb") as f:
        f.write(audio_bytes)

    size = request.size or DEFAULT_SIZE
    save_file = os.path.join(tmp_dir, "output.mp4")

    cmd = [
        sys.executable,
        os.path.join(WAN_PATH, "generate.py"),
        "--task", "s2v-14B",
        "--size", size,
        "--ckpt_dir", CKPT_DIR,
        "--offload_model", "True",
        "--convert_model_dtype",
        "--image", DEFAULT_ANCHOR,
        "--audio", wav_path,
        "--save_file", save_file,
    ]

    process = subprocess.Popen(cmd, cwd=WAN_PATH, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    process.communicate(timeout=INFERENCE_TIMEOUT)

    final_mp4 = os.path.join(tmp_dir, "final.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-i", save_file,
         "-c:v", "libx264", "-crf", "23", "-preset", "fast",
         "-c:a", "aac", "-b:a", "128k",
         "-movflags", "+faststart", final_mp4],
        check=True, capture_output=True,
    )

    with open(final_mp4, "rb") as f:
        video_bytes = f.read()

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return Response(content=video_bytes, media_type="video/mp4")
