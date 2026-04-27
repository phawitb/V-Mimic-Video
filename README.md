# Mimic-Videos

ระบบอัตโนมัติสำหรับแปลงวิดีโอ TikTok ภาษาอังกฤษ ให้เป็นวิดีโอพากย์ไทยพร้อมซับไตเติ้ล ในรูปแบบ TikTok แนวตั้ง (1080x1920) พร้อม Title Hook, เสียงพากย์ไทย, เพลง Background และเอฟเฟกต์อนิเมะ (เลือกได้)

---

## สารบัญ

- [ภาพรวมระบบ](#ภาพรวมระบบ)
- [โครงสร้างโปรเจกต์](#โครงสร้างโปรเจกต์)
- [ความต้องการของระบบ](#ความต้องการของระบบ)
- [การติดตั้ง](#การติดตั้ง)
- [การรัน](#การรัน)
- [Pipeline 7 ขั้นตอน](#pipeline-7-ขั้นตอน)
- [ระบบ Resumable](#ระบบ-resumable-รันต่อเมื่อ-fail)
- [โครงสร้าง Workspace](#โครงสร้าง-workspace)
- [Web Dashboard](#web-dashboard-apippy)
- [API Endpoints](#api-endpoints)
- [การตั้งค่า](#การตั้งค่า)
- [ตัวอย่างผลลัพธ์](#ตัวอย่างผลลัพธ์)

---

## ภาพรวมระบบ

```
TikTok URL (ภาษาอังกฤษ)
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Step 1: Download Video         (yt-dlp)             │
│  Step 2: Transcribe & Translate (Whisper + Google)    │
│  Step 3: Separate Background    (Demucs)              │
│  Step 4: Thai Voice Dubbing     (Edge-TTS / VoxCPM)   │
│  Step 5: Anime Style (optional) (AnimeGAN2)           │
│  Step 6: AI Title & Tags        (Ollama/Llama3)       │
│  Step 7: Final Composition      (MoviePy + OpenCV)    │
└──────────────────────────────────────────────────────┘
    │
    ▼
TikTok-Ready Video (1080x1920)
พากย์ไทย + ซับไทย + Background Music + Title Hook
```

---

## โครงสร้างโปรเจกต์

```
Mimic-Videos/
├── app.py              # FastAPI Web Server + Job Queue Manager
├── run_all.py          # Core Pipeline (7 Steps)
├── index.html          # React Web Dashboard (Single-file)
├── mhee_voice.mp3      # เสียงอ้างอิงสำหรับ VoxCPM Voice Cloning
├── state.json          # สถานะ Queue/Jobs (สร้างอัตโนมัติ)
├── fonts/
│   ├── FkLindoBold.ttf
│   ├── FkLindoRegular.ttf
│   ├── FkLindoThin.ttf
│   └── สำหรับCanva/
│       └── FkLindoBoldCV.ttf   # ฟอนต์หลักสำหรับ Title & Subtitle
├── ws/                 # Workspace ชั่วคราว (สร้างอัตโนมัติ)
│   └── tiktok_{id}/    # ไฟล์ระหว่างประมวลผลของแต่ละ Job
├── outputs/            # ผลลัพธ์วิดีโอสุดท้าย (สร้างอัตโนมัติ)
│   └── {playlist}/     # จัดกลุ่มตาม Playlist
│       ├── tk_tiktok_{id}{suffix}.mp4
│       └── tk_tiktok_{id}.json
└── README.md
```

---

## ความต้องการของระบบ

### System Requirements

| รายการ | ขั้นต่ำ |
|--------|---------|
| OS | macOS (Apple Silicon M1/M2/M3+), Linux, Windows |
| Python | 3.9+ |
| RAM | 8 GB+ (แนะนำ 16 GB สำหรับ Anime + VoxCPM) |
| Disk | 2 GB+ ว่าง (ต่อ Job ใช้ ~500MB ระหว่างประมวลผล) |
| GPU | ไม่จำเป็น (ใช้ CPU ได้ แต่ CUDA จะเร็วกว่ามากสำหรับ Anime) |

### Software ที่ต้องติดตั้งก่อน

| Software | คำสั่งติดตั้ง (macOS) | ใช้ทำอะไร |
|----------|----------------------|-----------|
| **Python 3.9+** | `brew install python@3.11` | Runtime |
| **FFmpeg** | `brew install ffmpeg` | ตัดต่อวิดีโอ/เสียง |
| **ImageMagick** | `brew install imagemagick` | สร้าง TextClip สำหรับ Title/Subtitle |
| **Ollama** (optional) | `brew install ollama` | AI สร้าง Title Hook & Tags |

ตรวจสอบว่าติดตั้งแล้ว:

```bash
python3 --version    # Python 3.9+
ffmpeg -version      # FFmpeg
magick --version     # ImageMagick
ollama --version     # Ollama (optional)
```

---

## การติดตั้ง

### 1. Clone โปรเจกต์

```bash
git clone <repo-url>
cd Mimic-Videos
```

### 2. สร้าง Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
```

### 3. ติดตั้ง Python Dependencies

```bash
pip install fastapi uvicorn
pip install torch torchaudio torchvision
pip install openai-whisper
pip install edge-tts
pip install yt-dlp
pip install moviepy==1.0.3
pip install opencv-python-headless
pip install Pillow numpy
pip install pysubs2
pip install librosa soundfile
pip install deep-translator
pip install pythainlp
pip install demucs
pip install tqdm requests
```

> **VoxCPM (Voice Cloning, optional):**
> ```bash
> pip install voxcpm
> ```
> ต้องมีไฟล์ `mhee_voice.mp3` เป็นเสียงอ้างอิง

### 4. ตั้งค่า Ollama (optional - สำหรับ AI Title)

```bash
ollama pull llama3
ollama serve          # รันที่ localhost:11434
```

ถ้าไม่มี Ollama ระบบจะใช้ title default: `"สรุปคลิปที่น่าสนใจ"`

### 5. ตรวจสอบ Path ของ ImageMagick

ใน `run_all.py` บรรทัดที่ 32:

```python
MAGICK_PATH = "/opt/homebrew/bin/magick"
```

ถ้า path ไม่ตรง ให้หาด้วย:

```bash
which magick    # macOS/Linux
where magick    # Windows
```

แล้วแก้ค่า `MAGICK_PATH` ให้ตรง

### 6. ตรวจสอบ Font Path

ใน `run_all.py` บรรทัดที่ 33:

```python
FONT_PATH = "/Users/phawit/Projects/Mimic-Videos/fonts/สำหรับCanva/FkLindoBoldCV.ttf"
```

แก้ให้ตรงกับ path จริงของเครื่อง

---

## การรัน

### วิธีที่ 1: Web Dashboard (แนะนำ)

```bash
python app.py
```

เปิดเบราว์เซอร์ไปที่ **http://localhost:8123**

จะได้ Dashboard สำหรับ:
- เพิ่ม URL วิดีโอเข้า Queue
- ดู Log แบบ Real-time
- จัดการ Job (Run / Interrupt / Delete / Edit)
- ตั้งเวลาทำงาน (Working Hours)

### วิธีที่ 2: CLI โดยตรง

```bash
# รันแบบ default (Edge-TTS, ไม่มี Anime)
python run_all.py --url "https://www.tiktok.com/@user/video/1234567890" --playlist "my_playlist"

# รันพร้อม Anime Effect
python run_all.py --url "URL" --playlist "my_playlist" --anime

# รันพร้อม VoxCPM Voice Cloning
python run_all.py --url "URL" --playlist "my_playlist" --voxcpm

# รันทุกอย่าง
python run_all.py --url "URL" --playlist "my_playlist" --anime --voxcpm
```

| Flag | ค่า Default | คำอธิบาย |
|------|-------------|----------|
| `--url` | URL ใน run_all.py | TikTok Video URL |
| `--playlist` | `spoil_movie` | ชื่อ playlist (สร้างโฟลเดอร์ใน outputs/) |
| `--anime` | `False` | เปิดเอฟเฟกต์อนิเมะ |
| `--voxcpm` | `False` | ใช้ VoxCPM Voice Cloning แทน Edge-TTS |

---

## Pipeline 7 ขั้นตอน

### Step 1: Download Video

- ใช้ **yt-dlp** ดาวน์โหลดวิดีโอจาก TikTok
- รองรับ URL ย่อ (`vm.tiktok.com`, `vt.tiktok.com`) — resolve อัตโนมัติ
- Output: `ws/tiktok_{id}/input_video.mp4`

### Step 2: Transcribe & Translate

- **Whisper** (model: base) ถอดเสียงอังกฤษเป็นข้อความ พร้อม Timing
- **Google Translator** แปลแต่ละ segment เป็นภาษาไทย
- Output: `ws/tiktok_{id}/thai_sub.srt`

### Step 3: Separate Background Audio

- **Demucs** (Facebook) แยกเสียงเป็น vocals + no_vocals
- เก็บเฉพาะ `no_vocals.wav` (เพลง/เสียง background)
- Output: `ws/tiktok_{id}/separated/htdemucs/input_video/no_vocals.wav`

### Step 4: Generate Thai Dubbing

- สร้างเสียงพากย์ไทยจาก subtitle แต่ละบรรทัด
- **Edge-TTS** (default): ใช้เสียง `th-TH-NiwatNeural`
- **VoxCPM** (optional): Clone เสียงจาก `mhee_voice.mp3`
- ตัด silence จากแต่ละ clip + จัดเวลาไม่ให้ซ้อนกัน
- Output: `ws/tiktok_{id}/thai_dub.mp3` (เสียงพากย์ไทยอย่างเดียว)

### Step 5: Anime Style Conversion (optional)

- ใช้ **AnimeGAN2** (FacePaint v2 style) แปลงทุก frame
- ใช้ GPU ถ้ามี (CUDA), fallback เป็น CPU
- Output: `ws/tiktok_{id}/input_video_anime.mp4`

### Step 6: AI Content Suggestions (optional)

- ส่ง subtitle ให้ **Ollama/Llama3** วิเคราะห์
- สร้าง: `title`, `detail`, `tags`, `title_hook`
- retry สูงสุด 5 ครั้ง, fallback ถ้า Ollama ไม่พร้อม
- Output: `ws/tiktok_{id}/data.json`

### Step 7: Final Composition

สร้างวิดีโอ TikTok แนวตั้ง (1080x1920):

```
┌──────────────────────┐
│   TITLE BAR (แดง)     │  ← Title Hook (ฟอนต์ 95pt ขาว)
│   280px / 420px       │     สูง 280px (1 บรรทัด) หรือ 420px (2+ บรรทัด)
├──────────────────────┤
│                      │
│   VIDEO CONTENT      │  ← 1:1 Square Crop (1080x1080)
│   1080 x 1080        │     Smart Crop ตัด Black Bar อัตโนมัติ
│                      │
├──────────────────────┤
│   SUBTITLE BAR (ฟ้า)  │  ← ซับไทย (ฟอนต์ 62pt ดำ)
│   130px              │     ตัดคำอัตโนมัติ (pythainlp)
├──────────────────────┤
│   BLACK PADDING      │  ← เติมดำให้ครบ 1920px
└──────────────────────┘
```

- **เสียง:** mix Background Music (15% volume) + Thai Voice (120% volume)
- **Encoding:** H.264, AAC, 30fps
- Output: `outputs/{playlist}/tk_tiktok_{id}{suffix}.mp4`

---

## ระบบ Resumable (รันต่อเมื่อ Fail)

Pipeline ออกแบบให้ **รันต่อจาก step ที่ค้าง** ได้โดยไม่ต้องเริ่มใหม่:

### หลักการทำงาน

- แต่ละ step เช็คว่า output file **มีอยู่และสมบูรณ์** หรือไม่ก่อนรัน
- ถ้าสมบูรณ์ → ข้าม (แสดง ⏭️ ใน log)
- ถ้าไม่มี หรือไม่สมบูรณ์ (ไฟล์เสีย/ขนาด 0) → ทำใหม่

### Validation แต่ละประเภทไฟล์

| ประเภท | ฟังก์ชัน | เงื่อนไขผ่าน |
|--------|----------|-------------|
| Video (.mp4) | `is_valid_video()` | ขนาด > 1KB, OpenCV เปิดได้, มี frame > 0 |
| Audio (.wav/.mp3) | `is_valid_audio()` | ขนาด > 100B, librosa load ได้, มี audio data |
| Subtitle (.srt) | `is_valid_srt()` | ขนาด > 10B, pysubs2 parse ได้, มี entries > 0 |
| JSON (.json) | `is_valid_json()` | ขนาด > 5B, parse เป็น dict ได้, มี keys |

### ตัวอย่าง Scenario

```
ครั้งแรก: รันถึง Step 4 แล้ว fail (TTS error)
  Step 1: ✅ ดาวน์โหลดเสร็จ → input_video.mp4 มีอยู่
  Step 2: ✅ subtitle เสร็จ   → thai_sub.srt มีอยู่
  Step 3: ✅ แยกเสียงเสร็จ   → no_vocals.wav มีอยู่
  Step 4: ❌ fail

กด Run ใหม่:
  Step 1: ⏭️ ข้าม (video valid)
  Step 2: ⏭️ ข้าม (srt valid)
  Step 3: ⏭️ ข้าม (audio valid)
  Step 4: 🚀 ทำใหม่ (ไม่มี thai_dub.mp3)
  Step 5-7: 🚀 ทำต่อ
```

### Workspace Cleanup

- **ระหว่างทำงาน:** ไม่ลบ workspace — เก็บไว้เพื่อ resume
- **เสร็จสมบูรณ์:** ลบ `ws/tiktok_{id}/` อัตโนมัติหลังสร้าง final video สำเร็จ
- **Fail:** workspace ยังอยู่ → กด Run ใหม่จะต่อจาก step ที่ค้าง

---

## Web Dashboard (app.py)

### หน้าจอหลัก

เปิดที่ `http://localhost:8123` จะได้ Dashboard ที่มี:

- **Add Links:** วาง URL วิดีโอ (หลายรายการได้ ขึ้นบรรทัดใหม่)
- **Job List:** แสดงรายการ Job ทั้งหมดพร้อมสถานะ
- **Log Viewer:** แสดง Log แบบ Real-time ของ Job ที่กำลังทำ
- **Settings:** ตั้งเวลาทำงาน + เปิด/ปิด Queue

### สถานะของ Job

| สถานะ | คำอธิบาย |
|-------|----------|
| `queued` | รอคิว |
| `processing` | กำลังประมวลผล |
| `completed` | เสร็จสมบูรณ์ (มีลิงก์ดูวิดีโอ) |
| `failed` | ล้มเหลว (กด Run เพื่อลองใหม่ได้ — จะ resume จาก step ที่ค้าง) |

### ระบบ Auto-Retry

- Job ที่ fail จะถูก retry อัตโนมัติ **วันละ 1 ครั้ง** ในช่วง Working Hours
- ถ้า Queue ว่างแต่มี failed jobs → retry อัตโนมัติ
- Job ที่ `processing` ค้างตอน server restart → กลับเป็น `queued` อัตโนมัติ

---

## API Endpoints

| Method | Endpoint | คำอธิบาย |
|--------|----------|----------|
| `GET` | `/` | หน้า Dashboard (index.html) |
| `GET` | `/status` | สถานะระบบ (settings, queue count, active job) |
| `GET` | `/jobs` | รายการ Job ทั้งหมด (ไม่รวม logs) |
| `GET` | `/jobs/{id}/logs` | ดู logs ของ Job |
| `POST` | `/links` | เพิ่ม URL เข้า Queue |
| `POST` | `/jobs/{id}/run` | ย้าย Job ไปหน้า Queue / Retry |
| `POST` | `/jobs/{id}/interrupt` | หยุด Job ที่กำลังทำ |
| `PATCH` | `/jobs/{id}` | แก้ไข Job (playlist, anime, voxcpm) |
| `DELETE` | `/jobs/{id}` | ลบ Job |
| `POST` | `/settings` | อัปเดตเวลาทำงาน |
| `GET` | `/open-file?path=...` | เปิดไฟล์ด้วย OS default app |
| `GET` | `/open-folder?path=...` | เปิดโฟลเดอร์ใน Finder/Explorer |

### ตัวอย่าง: เพิ่ม Job ผ่าน API

```bash
curl -X POST http://localhost:8123/links \
  -H "Content-Type: application/json" \
  -d '{
    "video_urls": [
      "https://www.tiktok.com/@user/video/1234567890",
      "https://www.tiktok.com/@user/video/0987654321"
    ],
    "playlist_name": "spoil_movie",
    "use_voxcpm_tts": false,
    "use_anime": true
  }'
```

---

## การตั้งค่า

### ใน run_all.py (ค่า Default สำหรับ CLI)

```python
USE_VOXCPM_TTS = True     # ใช้ VoxCPM Voice Cloning
USE_ANIME = True           # เปิดเอฟเฟกต์อนิเมะ
PLAYLIST_NAME = "spoil_movie"
MAGICK_PATH = "/opt/homebrew/bin/magick"    # path ของ ImageMagick
FONT_PATH = "fonts/สำหรับCanva/FkLindoBoldCV.ttf"
```

### ใน Web Dashboard (Settings)

| ตั้งค่า | Default | คำอธิบาย |
|---------|---------|----------|
| Start Time | `09:00` | เวลาเริ่มทำงาน |
| End Time | `23:59` | เวลาหยุดทำงาน |
| Enabled | `true` | เปิด/ปิด Queue อัตโนมัติ |

### ปรับ Volume เสียง

ใน `create_final_tiktok_video()` ของ `run_all.py`:

```python
voice_clip = AudioFileClip(voice_audio).volumex(1.2)   # เสียงพากย์ไทย 120%
bg_clip = AudioFileClip(bg_audio_path).volumex(0.15)    # เพลง Background 15%
```

---

## ตัวอย่างผลลัพธ์

### ชื่อไฟล์ Output

รูปแบบ: `tk_tiktok_{video_id}{suffix}.mp4`

| ตัวเลือก | ชื่อไฟล์ |
|----------|----------|
| Default (Edge-TTS, ไม่มี Anime) | `tk_tiktok_7613651491974286614.mp4` |
| Anime only | `tk_tiktok_7613651491974286614_anime.mp4` |
| VoxCPM only | `tk_tiktok_7613651491974286614_voxcpm.mp4` |
| Anime + VoxCPM | `tk_tiktok_7613651491974286614_anime_voxcpm.mp4` |

### JSON Metadata

```json
{
    "title": "หนังระทึกขวัญ",
    "detail": "เรื่องราวของชายที่ติดอยู่ในห้องปริศนา",
    "tags": ["#สปอยหนัง", "#หนังระทึกขวัญ", "#TikTok"],
    "title_hook": "ถ้าคุณออกจากห้องนี้ไม่ได้ คุณจะทำอย่างไร?"
}
```

---

## Troubleshooting

| ปัญหา | วิธีแก้ |
|-------|---------|
| `ImageMagick not found` | ตรวจสอบ `MAGICK_PATH` ใน run_all.py ว่าตรงกับ `which magick` |
| `FFmpeg not found` | `brew install ffmpeg` |
| Subtitle ภาษาไทยเพี้ยน | ตรวจสอบว่า font file มีอยู่จริงตาม `FONT_PATH` |
| Video download ล้มเหลว | อัปเดต yt-dlp: `pip install -U yt-dlp` |
| Ollama ไม่ตอบ | ตรวจสอบว่า `ollama serve` รันอยู่ และ `ollama pull llama3` แล้ว |
| เสียง Background เบาเกินไป | เพิ่ม volume ใน `create_final_tiktok_video()`: `.volumex(0.15)` → `.volumex(0.3)` |
| Anime ช้ามาก | ปกติถ้าไม่มี GPU — ใช้เวลาหลายนาทีต่อวิดีโอ |
| Port 8123 ถูกใช้แล้ว | แก้ port ใน `app.py` บรรทัดสุดท้าย: `uvicorn.run(..., port=XXXX)` |
