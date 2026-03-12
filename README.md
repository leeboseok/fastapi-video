# SadTalker Video Server

SadTalker 기반 립싱크 영상 생성 + FFmpeg 뉴스 템플릿 합성 API 서버입니다.

## 기술 스택

- **립싱크 엔진**: [SadTalker](https://github.com/OpenTalker/SadTalker)
- **영상 합성**: FFmpeg (H.264 + AAC, 480p 뉴스 템플릿)
- **API 프레임워크**: FastAPI + Uvicorn
- **Python**: 3.10

## Docker 실행

### 이미지 빌드

```bash
docker build -t video-server .
```

### 컨테이너 실행

```bash
docker run -d -p 11113:11113 --name video-server video-server
```

### GPU가 필요한 경우 (NVIDIA):
```bash
docker run -d -p 11113:11113 --gpus all --name video-server video-server
docker build -t video-server . && docker stop video-server && docker rm video-server && docker run -d --name video-server --gpus all -p 11113:11113 video-server
```