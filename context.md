# Wan2.2-S2V-14B 마이그레이션 컨텍스트

> 작성일: 2026-03-12
> 목적: SadTalker → Wan2.2-S2V-14B 교체를 위한 전체 컨텍스트 정리

---

## 1. 프로젝트 배경

### 기존 프로젝트 (SadTalker 기반)
- **저장소**: `git@github.com:leeboseok/python-video-server.git` (branch: `dev`)
- **구조**: FastAPI 마이크로서비스 (단일 파일 `app/main.py`, 103줄)
- **기능**: Base64 오디오 입력 → SadTalker 립싱크 → FFmpeg 인코딩 → MP4 응답
- **앱 타이틀**: "ARMS API"
- **포트**: 11113

### 기존 API 엔드포인트
| Method | Path | 기능 |
|--------|------|------|
| GET | `/` | 헬스체크 (`{"status": "ok"}`) |
| POST | `/generate-video` | 오디오(base64) → 영상 생성 |

### 기존 요청/응답 형식
- **Request**: `{ "audioData": "<base64 encoded audio>" }`
- **Response**: `video/mp4` 파일 스트리밍
- **에러**: 400 (빈 오디오, base64 실패), 504 (타임아웃), 500 (생성 실패)

### 기존 SadTalker 파라미터
- Source image: `/app/resources/img.png` (고정 앵커 이미지)
- Preprocessing: resize 512x512
- Expression scale: 0.5
- Timeout: 10,800초 (3시간)
- FFmpeg: H.264 (libx264, CRF 23, fast preset), AAC 128k, faststart

---

## 2. 인프라 환경

| 항목 | 값 |
|------|-----|
| **클라우드** | AWS EC2 Ubuntu |
| **GPU** | H100 80GB (영상 생성 시에만 동적 연결) |
| **배포** | Docker (CPU 베이스) 상시 가동 → GPU 필요 시 H100 연결 |
| **포트** | 11113 |
| **로컬 테스트** | RTX 5070 (12GB VRAM, `--offload_model` 필요) |
| **Dockerfile** | CPU 베이스 이미지 (GPU 드라이버는 호스트에서 제공) |

### 운영 구조
```
EC2 (상시): Docker(CPU) → FastAPI 서버 대기
영상 생성 요청 시: H100 동적 연결 → 추론 → 완료 후 GPU 해제
```
- GPU 동적 연결 메커니즘은 이미 EC2에 구성 완료
- 비용 최적화: GPU 사용 시간만큼만 과금

---

## 3. Wan2.2-S2V-14B 모델 상세

### 기본 정보
| 항목 | 값 |
|------|-----|
| **개발** | Alibaba Tongyi Lab (HumanAIGC Team) |
| **아키텍처** | Mixture-of-Experts (MoE) Diffusion Model |
| **파라미터** | 27B total / 14B active per step |
| **라이선스** | Apache 2.0 |
| **논문** | arXiv:2508.18621 |
| **저장소** | https://github.com/Wan-Video/Wan2.2 |
| **모델** | https://huggingface.co/Wan-AI/Wan2.2-S2V-14B |

### MoE 구조
- High-noise expert: 초기 디노이징 단계 (전체 레이아웃 결정)
- Low-noise expert: 후기 디노이징 단계 (디테일 보정)
- SNR(Signal-to-Noise Ratio) 기반 전환

### 입출력
- **Input**: 오디오(WAV/MP3) + 참조 이미지(JPG/PNG) + 텍스트 프롬프트(선택) + 포즈 비디오(선택)
- **Output**: 480P / 720P @ 24fps 영상
- **영상 길이**: 오디오 길이에 자동 맞춤 (`--num_clip` 미지정 시)

### 핵심 기능
- 오디오 기반 표정 + 신체 동작 생성 (대화, 노래, 퍼포먼스)
- 시네마틱 카메라 워크
- 포즈 기반 생성 (pose video 입력, 선택)
- 반신/전신 지원
- 실사, 만화, 동물, 디지털 휴먼 지원
- FramePack 기술로 장시간 영상 메모리 문제 해결

### 벤치마크 성능
| Metric | Wan2.2-S2V | 의미 |
|--------|-----------|------|
| FID | 15.66 (Best) | 영상 품질 |
| EFID | 0.283 (Best) | 표정 진정성 |
| CSIM | 0.677 (Best) | 아이덴티티 일관성 |
| SSIM | 0.734 (Best) | 구조적 유사도 |
| PSNR | 20.49 (Best) | 화질 |

비교 대상: EchoMimicV2, MimicMotion, EMO2, FantasyTalking, Hunyuan-Avatar

---

## 4. Wan2.2-S2V 실행 방법

### 설치
```bash
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
pip install -r requirements.txt
pip install -r requirements_s2v.txt
```

### 모델 다운로드
```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir ./Wan2.2-S2V-14B
```

### 의존성 (requirements.txt)
```
torch>=2.4.0
torchvision>=0.19.0
torchaudio
opencv-python>=4.9.0.80
diffusers>=0.31.0
transformers>=4.49.0,<=4.51.3
tokenizers>=0.20.3
accelerate>=1.1.1
tqdm
imageio[ffmpeg]
easydict
ftfy
dashscope
imageio-ffmpeg
flash_attn
numpy>=1.23.5,<2
```
+ `requirements_s2v.txt` (S2V 전용 의존성)

### Single GPU 추론 (H100 80GB)
```bash
python generate.py \
  --task s2v-14B \
  --size 1024*704 \
  --ckpt_dir ./Wan2.2-S2V-14B/ \
  --prompt "description of the scene" \
  --image "input_image.jpg" \
  --audio "input_audio.wav"
```
> H100 80GB에서는 `--offload_model`, `--convert_model_dtype`, `--t5_cpu` 불필요 (풀 스피드)

### Multi-GPU 추론 (8x GPU)
```bash
torchrun --nproc_per_node=8 generate.py \
  --task s2v-14B \
  --size 1024*704 \
  --ckpt_dir ./Wan2.2-S2V-14B/ \
  --dit_fsdp --t5_fsdp --ulysses_size 8 \
  --prompt "..." --image "..." --audio "..."
```

### 주요 CLI 파라미터
| 파라미터 | 설명 |
|----------|------|
| `--task` | `s2v-14B` |
| `--size` | 생성 해상도 (예: `1024*704`) |
| `--ckpt_dir` | 모델 체크포인트 경로 |
| `--prompt` | 텍스트 프롬프트 (선택) |
| `--image` | 참조 이미지 경로 |
| `--audio` | 오디오 파일 경로 |
| `--pose_video` | 포즈 참조 비디오 (선택) |
| `--num_clip` | 클립 수 (미지정 시 오디오 길이에 자동 맞춤) |
| `--offload_model` | CPU 오프로드 (H100에서는 불필요) |
| `--enable_tts` | TTS 활성화 (CosyVoice) |

---

## 5. SadTalker vs Wan2.2-S2V 비교

| 항목 | SadTalker | Wan2.2-S2V-14B |
|------|-----------|----------------|
| 생성 범위 | 얼굴/머리만 | 반신/전신 + 카메라 워크 |
| 품질 | 중간 (uncanny valley) | 시네마틱 수준 |
| 표정 | 립싱크 + 기본 표정 | 정교한 표정 + 신체 동작 |
| 해상도 | 256x256 | 480P / 720P |
| VRAM | 4-8GB | 80GB (H100 최적) |
| 추론 속도 | 수 분 | 수십 분 ~ 1시간+ (3분 영상) |
| 아키텍처 | 3DMM 경량 | 27B MoE Diffusion |
| 라이선스 | MIT | Apache 2.0 |

---

## 6. 마이그레이션 설계 가이드

### 목표
- SadTalker를 Wan2.2-S2V-14B로 완전 교체
- 동일한 API 인터페이스 유지 (POST `/generate-video`)
- Docker + H100 GPU 환경
- 3분 내외 영상 생성 지원

### 변경 포인트

#### Dockerfile
- Base image: `nvidia/cuda:11.8.0` → CUDA 12.x + PyTorch 2.4+ 호환 이미지
- SadTalker clone/setup → Wan2.2 clone + `requirements.txt` + `requirements_s2v.txt`
- SadTalker 모델 다운로드 → `huggingface-cli download Wan-AI/Wan2.2-S2V-14B`
- 모델 크기 대폭 증가 (~50GB+), Docker 이미지 크기 주의

#### main.py
- SadTalker `inference.py` subprocess 호출 → Wan2.2 `generate.py` 호출로 변경
- 프롬프트 파라미터 추가 고려 (`--prompt`)
- 출력 파일 경로 변경 (Wan2.2의 출력 디렉토리 구조 확인 필요)
- timeout 값 상향 검토 (3분 영상 → 1시간+ 소요 가능)

#### FFmpeg 후처리
- 기존 FFmpeg 인코딩 로직 그대로 유지 가능
- Wan2.2 출력 파일 형식/경로만 맞추면 됨

#### API 변경 (선택)
- `GenerateVideoRequest`에 `prompt` 필드 추가 가능
- 기존 `audioData` 필드는 그대로 유지

### 추론 시간 참고
| 조건 | 예상 시간 |
|------|----------|
| H100 1대, 720P, 3분 영상 | 30분 ~ 1시간+ |
| H100 8대 (FSDP), 720P, 3분 영상 | ~10-20분 |
| H100 1대, 480P, 3분 영상 | 더 빠름 (미측정) |

---

## 7. 참고 자료

- HuggingFace 모델: https://huggingface.co/Wan-AI/Wan2.2-S2V-14B
- GitHub 저장소: https://github.com/Wan-Video/Wan2.2
- 논문 (Wan-S2V): https://arxiv.org/abs/2508.18621
- 논문 (Wan2.2 Base): https://arxiv.org/abs/2503.20314
- 프로젝트 페이지: https://humanaigc.github.io/wan-s2v-webpage/
- VRAM 가이드: https://blogs.novita.ai/wan-2-2-vram-find-the-best-gpu-setup-for-deployment/
- H100 최적화: https://www.voltagepark.com/blog/accelerating-wan2-2-from-4-67s-to-1-5s-per-denoising-step-through-targeted-optimizations
- fal.ai API: https://fal.ai/models/fal-ai/wan/v2.2-14b/speech-to-video
- Replicate: https://replicate.com/wan-video/wan-2.2-s2v/readme
