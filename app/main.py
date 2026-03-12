import os
import sys
import base64
import shutil
import glob
import subprocess
import tempfile
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ARMS API")

WAN_PATH = os.getenv("WAN_PATH", "/opt/Wan2.2")
CKPT_DIR = os.getenv("CKPT_DIR", "/opt/models/Wan2.2-S2V-14B")
DEFAULT_ANCHOR = os.getenv("DEFAULT_ANCHOR", "/app/resources/img.png")
DEFAULT_SIZE = os.getenv("DEFAULT_SIZE", "832*480")
DEFAULT_PROMPT = os.getenv(
    "DEFAULT_PROMPT",
    "A person is talking naturally with gentle expressions and subtle body movements.",
)
INFERENCE_TIMEOUT = int(os.getenv("INFERENCE_TIMEOUT", "21600"))  # 6h


class GenerateVideoRequest(BaseModel):
    audioData: str
    prompt: Optional[str] = None
    size: Optional[str] = None


@app.get("/")
def health_check():
    model_exists = os.path.isdir(CKPT_DIR)
    return {"status": "ok", "model_loaded": model_exists}


@app.post("/generate-video")
def generate_video(request: GenerateVideoRequest):
    if not request.audioData or not request.audioData.strip():
        raise HTTPException(status_code=400, detail="audioData가 비어있습니다.")

    if not os.path.isdir(CKPT_DIR):
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    tmp_dir = tempfile.mkdtemp(prefix="wan_video_")
    logger.info("Processing request in %s", tmp_dir)

    try:
        # 1. Base64 오디오 디코딩
        try:
            audio_bytes = base64.b64decode(request.audioData)
        except Exception:
            raise HTTPException(status_code=400, detail="Base64 디코딩 실패")

        wav_path = os.path.join(tmp_dir, "input.wav")
        with open(wav_path, "wb") as f:
            f.write(audio_bytes)

        # 2. Wan2.2-S2V 추론
        size = request.size or DEFAULT_SIZE
        prompt = request.prompt or DEFAULT_PROMPT

        cmd = [
            sys.executable,
            os.path.join(WAN_PATH, "generate.py"),
            "--task", "s2v-14B",
            "--size", size,
            "--ckpt_dir", CKPT_DIR,
            "--image", DEFAULT_ANCHOR,
            "--audio", wav_path,
            "--prompt", prompt,
        ]

        logger.info("Running Wan2.2: %s", " ".join(cmd))

        try:
            process = subprocess.Popen(
                cmd,
                cwd=tmp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            stdout, _ = process.communicate(timeout=INFERENCE_TIMEOUT)
            logger.info("Wan2.2 output:\n%s", stdout.decode(errors="replace"))
        except subprocess.TimeoutExpired:
            process.kill()
            raise HTTPException(
                status_code=504, detail="Wan2.2 처리 시간 초과"
            )

        if process.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Wan2.2 추론 실패 (exit code: {process.returncode})",
            )

        # 3. 출력 영상 탐색 (CWD 및 하위 디렉토리)
        mp4_files = glob.glob(os.path.join(tmp_dir, "**", "*.mp4"), recursive=True)
        if not mp4_files:
            raise HTTPException(status_code=500, detail="Wan2.2 출력 영상 없음")

        raw_mp4 = sorted(mp4_files, key=os.path.getmtime)[-1]
        logger.info("Found output: %s", raw_mp4)

        # 4. FFmpeg H.264 + AAC 후처리
        final_mp4 = os.path.join(tmp_dir, "final.mp4")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", raw_mp4,
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            final_mp4,
        ]

        try:
            subprocess.run(
                ffmpeg_cmd, timeout=3600, check=True,
                capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error("FFmpeg error: %s", e.stderr)
            raise HTTPException(status_code=500, detail="FFmpeg 인코딩 실패")
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="FFmpeg 처리 시간 초과")

        # 5. 응답
        with open(final_mp4, "rb") as f:
            video_bytes = f.read()

        logger.info("Returning video (%d bytes)", len(video_bytes))
        return Response(content=video_bytes, media_type="video/mp4")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
