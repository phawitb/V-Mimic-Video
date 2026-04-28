import os
import sys
import asyncio
import shutil
import torch
import torchaudio
import numpy as np
import cv2
import json
import requests
import whisper
import pysubs2
import edge_tts
import yt_dlp
import soundfile as sf
import librosa
from PIL import Image
from tqdm import tqdm
from moviepy.editor import (VideoFileClip, AudioFileClip, CompositeAudioClip, 
                            TextClip, CompositeVideoClip, ColorClip)
from moviepy.config import change_settings
from moviepy.video.tools.subtitles import SubtitlesClip
from pythainlp.tokenize import word_tokenize

# --- 0. CONFIGURATION & MAC M2 PATCHES ---
USE_VOXCPM_TTS = True
USE_ANIME = True  # เปลี่ยนเป็น False หากไม่ต้องการทำเอฟเฟกต์อนิเมะ
PLAYLIST_NAME = "spoil_movie"
# VIDEO_URL = "https://www.tiktok.com/@michaelmovies5/video/7610098188397071630"
VIDEO_URL = "https://www.tiktok.com/@chriefbegliiya2/video/7613651491974286614"

MAGICK_PATH = "/opt/homebrew/bin/magick"
FONT_PATH = "/Users/phawit/Projects/Mimic-Videos/fonts/สำหรับCanva/FkLindoBoldCV.ttf"
# "fonts/FkLindoBold.ttf"
WS_DIR = "ws"
OUTPUT_DIR = f"outputs/{PLAYLIST_NAME}"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["OBJC_DISABLE_INITIALIZE_FOR_SAFETY"] = "YES"
# ป้องกัน SDL2 Duplicate Error บน Mac
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

change_settings({"IMAGEMAGICK_BINARY": MAGICK_PATH})

def simple_save(filepath, src, sample_rate, encoding=None, bits_per_sample=None):
    data = src.t().cpu().numpy()
    sf.write(filepath, data, sample_rate)
torchaudio.save = simple_save

# รองรับ MoviePy speedx
try:
    from moviepy.audio.fx.all import speedx
except ImportError:
    from moviepy.video.fx.speedx import speedx

# --- 1. DIRECTORY SETUP ---
os.makedirs(WS_DIR, exist_ok=True)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ดึง ID ของวิดีโอมาเป็นชื่อไฟล์
video_id = VIDEO_URL.split('/')[-1].split('?')[0]
ENG_NAME = f"tiktok_{video_id}"

RAW_VIDEO = os.path.join(WS_DIR, "input_video.mp4")
ANIME_VIDEO = os.path.join(WS_DIR, "input_video_anime.mp4")
THAI_SUB = os.path.join(WS_DIR, "thai_sub.srt")
THAI_VOICE = os.path.join(WS_DIR, "thai_dub.mp3")
DATA_JSON = os.path.join(WS_DIR, "data.json")

# --- VALIDATION HELPERS ---

def is_valid_video(path):
    """Check if video file exists and is playable (has frames)."""
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        return False
    try:
        cap = cv2.VideoCapture(path)
        ok = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
        cap.release()
        return ok
    except:
        return False

def is_valid_audio(path):
    """Check if audio file exists and has duration > 0."""
    if not os.path.exists(path) or os.path.getsize(path) < 100:
        return False
    try:
        y, sr = librosa.load(path, sr=None, duration=1)
        return len(y) > 0
    except:
        return False

def is_valid_srt(path):
    """Check if SRT file exists and has subtitle entries."""
    if not os.path.exists(path) or os.path.getsize(path) < 10:
        return False
    try:
        subs = pysubs2.load(path)
        return len(subs) > 0
    except:
        return False

def is_valid_json(path):
    """Check if JSON file exists and is parseable with content."""
    if not os.path.exists(path) or os.path.getsize(path) < 5:
        return False
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return isinstance(data, dict) and len(data) > 0
    except:
        return False

# --- 2. STEP FUNCTIONS ---

def download_video_to_path(url, output_path):
    print(f"🔍 Analyzing: {url}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Sec-Fetch-Mode': 'navigate',
        },
        'merge_output_format': 'mp4',
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        'postprocessor_args': [
            '-vcodec', 'libx264',
            '-acodec', 'aac',
            '-pix_fmt', 'yuv420p'
        ],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                return {"status": "error", "message": "Failed to extract video info (None)"}
            
            extractor = info.get('extractor_key', 'Unknown').lower()
            return {
                "status": "success",
                "platform": extractor,
                "video_path": os.path.abspath(output_path),
                "title": info.get('title', 'Unknown Title')
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def detect_content_area(video_path):
    """Detects the content area of a video by sampling frames to remove black bars."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samples = np.linspace(0, total_frames - 1, 10, dtype=int)
    
    h, w = None, None
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0
    
    for s in samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s))
        ret, frame = cap.read()
        if not ret: continue
        if h is None: h, w = frame.shape[:2]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, fw, fh = cv2.boundingRect(coords)
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x + fw), max(max_y, y + fh)
    
    cap.release()
    if h is None or max_x == 0: return None
    return (min_x, min_y, max_x, max_y)

def generate_thai_sub_to_path(video_path, srt_path, ws_dir, log_func=None):
    def log(m): 
        if log_func: log_func(m)
        else: print(m)
        
    log("🚀 [Whisper] Loading model (base)...")
    model = whisper.load_model("base")
    log("🚀 [Whisper] Transcribing English audio...")
    result = model.transcribe(video_path, language="en")
    
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source='en', target='th')
    
    log(f"🚀 [Translator] Translating {len(result['segments'])} segments to Thai...")
    for i, segment in enumerate(result['segments']):
        try:
            translated = translator.translate(segment['text'])
            segment['text'] = translated
            if i % 5 == 0: log(f"   - Translated {i+1}/{len(result['segments'])}...")
        except Exception as e:
            log(f"   - Translation error at segment {i}: {e}")
            
    writer = whisper.utils.get_writer("srt", ws_dir)
    base_name = os.path.basename(video_path).replace(".mp4", "")
    writer(result, base_name)
    
    generated_srt = os.path.join(ws_dir, f"{base_name}.srt")
    if os.path.exists(generated_srt) and generated_srt != srt_path:
        if os.path.exists(srt_path): os.remove(srt_path)
        os.rename(generated_srt, srt_path)
    log("✅ Subtitles ready.")

def separate_bg_audio_to_path(video_path, ws_dir):
    print("🚀 Separating Background (Demucs)...")
    from demucs.separate import main as demucs_main
    current_dir = os.getcwd()
    os.chdir(ws_dir) 
    video_name = os.path.basename(video_path)
    sys.argv = ["demucs", "--two-stems", "vocals", video_name]
    try:
        demucs_main()
    except SystemExit: pass 
    finally: os.chdir(current_dir)
    return os.path.join(ws_dir, f"separated/htdemucs/{video_name.replace('.mp4', '')}/no_vocals.wav")

def trim_silence_from_array(y, top_db=25):
    """Trims leading and trailing silence from an audio array."""
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    return yt

async def make_final_audio_to_path(srt_path, output_audio, ws_dir, use_voxcpm, log_func=None):
    def log(m):
        if log_func: log_func(m)
        else: print(m)

    log("🚀 [TTS] Loading subtitles...")
    subs = pysubs2.load(srt_path)
    temp_voices_dir = os.path.join(ws_dir, "temp_voices")
    if os.path.exists(temp_voices_dir): shutil.rmtree(temp_voices_dir)
    os.makedirs(temp_voices_dir)

    dub_clips = []
    num_subs = len(subs)

    log(f"🚀 [TTS] Generating {num_subs} voice clips (Natural Speed)...")

    vox_model = None
    if use_voxcpm:
        from voxcpm import VoxCPM
        log("🚀 [VoxCPM] Loading pre-trained model...")
        vox_model = VoxCPM.from_pretrained("openbmb/VoxCPM2")

    last_end_time = 0.0

    for i, line in enumerate(subs):
        if not line.text.strip(): continue

        temp_mp3 = os.path.join(temp_voices_dir, f"v_{i}.mp3")
        sample_rate = 24000 # Default for edge-tts

        if not use_voxcpm:
            communicate = edge_tts.Communicate(line.text, "th-TH-NiwatNeural")
            await communicate.save(temp_mp3)
            # Load and trim
            y, sr = librosa.load(temp_mp3, sr=None)
            y_trimmed = trim_silence_from_array(y)
            sf.write(temp_mp3, y_trimmed, sr)
        else:
            mp3_data = vox_model.generate(
                text=line.text,
                reference_wav_path="mhee_voice.mp3",
                cfg_value=2.5,
                inference_timesteps=12,
                normalize=True
            )
            sample_rate = vox_model.tts_model.sample_rate
            y_trimmed = trim_silence_from_array(mp3_data)
            sf.write(temp_mp3, y_trimmed, sample_rate)

        if i % 10 == 0: log(f"   - Generated & Trimmed {i+1}/{num_subs} voices...")

        audio_clip = AudioFileClip(temp_mp3).volumex(1.0)

        # Calculate start time with "push" logic to avoid overlaps
        intended_start = line.start / 1000.0
        # If the intended start is earlier than the last clip ended, push it to last_end_time
        actual_start = max(intended_start, last_end_time)

        audio_clip = audio_clip.set_start(actual_start)
        dub_clips.append(audio_clip)

        # Update last_end_time for the next clip
        last_end_time = actual_start + audio_clip.duration

    log("🚀 [Mixer] Combining Thai voice dubs...")
    final_mix = CompositeAudioClip(dub_clips)
    final_mix.write_audiofile(output_audio, fps=44100, logger=None)
    log(f"✅ Thai voice audio complete. Total duration: {final_mix.duration:.2f}s")

def run_anime_conversion_to_path(input_video, output_video, log_func=None):
    def log(m): 
        if log_func: log_func(m)
        else: print(m)

    log("🎨 [Anime] Initializing AI models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2").to(device).eval()
    face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", size=512, device=device)

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    log(f"🎨 [Anime] Processing {total_frames} frames (Style: FacePaint v2)...")
    with torch.no_grad():
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            anime_pil = face2paint(model, pil_img, size=512)
            anime_frame = cv2.cvtColor(np.array(anime_pil), cv2.COLOR_RGB2BGR)
            if anime_frame.shape[1] != width or anime_frame.shape[0] != height:
                anime_frame = cv2.resize(anime_frame, (width, height))
            out.write(anime_frame)
            if i % 100 == 0: log(f"   - Rendered frame {i}/{total_frames} ({(i/total_frames)*100:.1f}%)")
    cap.release()
    out.release()
    log("✅ Anime conversion complete.")

def get_ollama_suggestions_to_path(srt_path, data_json_path, log_func=None):
    def log(m): 
        if log_func: log_func(m)
        else: print(m)

    log("🤖 [AI] Getting content suggestions from Ollama...")
    if not os.path.exists(srt_path): return {}
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    srt_content = " ".join([l.strip() for l in lines if l.strip() and not l.strip().isdigit() and '-->' not in l])
    
    url = "http://localhost:11434/api/generate"
    prompt = f"""
    วิเคราะห์เนื้อหาซับไตเติ้ลวิดีโอต่อไปนี้:
    {srt_content[:3000]} 
    
    จงสรุปเนื้อหาสำหรับนำไปทำ Content TikTok โดยตอบกลับเป็นรูปแบบ JSON ภาษาไทยเท่านั้น
    โครงสร้าง Keys ที่ต้องการ:
    1. "title": หัวข้อหลักของวิดีโอ
    2. "detail": สรุปรายละเอียดสั้นๆ (1 ประโยค)
    3. "tags": Hashtags ที่เกี่ยวข้อง 3-5 อัน
    4. "title_hook": ข้อความพาดหัวดึงดูดใจ (ไม่เกิน 1 บรรทัด)
    
    ตอบเฉพาะ JSON เท่านั้น ห้ามมีข้อความเกริ่นนำ
    """
    
    payload = {"model": "llama3", "prompt": prompt, "format": "json", "stream": False}
    
    try:
        response = requests.post(url, json=payload)
        res_json = json.loads(response.json()['response'])
        with open(data_json_path, "w", encoding="utf-8") as file:
            json.dump(res_json, file, indent=4, ensure_ascii=False)
        return res_json
    except: return {"title_hook": "สรุปเนื้อหาวิดีโอที่น่าสนใจ"}

def resolve_tiktok_url(url):
    """Resolves shortened TikTok URLs to their full version to extract the real video ID."""
    try:
        if "vt.tiktok.com" in url or "vm.tiktok.com" in url:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            # Use a session to handle redirects efficiently
            with requests.Session() as session:
                response = session.get(url, allow_redirects=True, timeout=10, headers=headers)
                return response.url
    except Exception as e:
        print(f"⚠️ Warning: Could not resolve URL {url}: {e}")
    return url

# --- 3. VIDEO COMPOSITING FUNCTIONS ---

def wrap_thai(text, width_limit=22):
    words = word_tokenize(text, engine="newmm")
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) > width_limit:
            lines.append(cur); cur = w
        else: cur += w
    lines.append(cur)
    wrapped = "\n" + "\n".join(lines)
    return wrapped, len(lines)

def split_thai_sub(text, max_chars=38):
    words = word_tokenize(text, engine="newmm")
    l1, idx, cur_len = [], 0, 0
    for i, w in enumerate(words):
        if cur_len + len(w) <= max_chars:
            l1.append(w); cur_len += len(w); idx = i + 1
        else: break
    return "".join(l1), "".join(words[idx:])

def measure_rms(audio_path, duration=30):
    """Measure RMS amplitude of an audio file (sample up to `duration` seconds)."""
    y, sr = librosa.load(audio_path, sr=None, mono=True, duration=duration)
    rms = np.sqrt(np.mean(y ** 2))
    return float(rms)

def measure_active_rms(audio_path, duration=30, top_db=30):
    """Measure RMS of only the non-silent (active speech) portions of an audio file.
    This avoids the silence gaps in TTS audio dragging down the RMS value."""
    y, sr = librosa.load(audio_path, sr=None, mono=True, duration=duration)
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return float(np.sqrt(np.mean(y ** 2)))
    active = np.concatenate([y[s:e] for s, e in intervals])
    if len(active) == 0:
        return float(np.sqrt(np.mean(y ** 2)))
    return float(np.sqrt(np.mean(active ** 2)))

def mix_audio_with_ffmpeg(voice_path, bg_path, output_path, bg_vol, log_func=None):
    """Mix voice and background audio using ffmpeg directly (more reliable than MoviePy)."""
    import subprocess
    def log(m):
        if log_func: log_func(m)
        else: print(m)

    if bg_path and os.path.exists(bg_path):
        # amix: mix voice (full volume) + bg (bg_vol scaled), duration=first (match voice length)
        # loudnorm on bg to normalize peaks before mixing
        cmd = [
            "ffmpeg", "-y",
            "-i", voice_path,
            "-i", bg_path,
            "-filter_complex",
            f"[1:a]volume={bg_vol:.4f}[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[out]",
            "-map", "[out]",
            "-acodec", "aac", "-b:a", "192k",
            output_path
        ]
        log(f"🎬 [ffmpeg-mixer] voice + bg (vol={bg_vol:.3f}) → {os.path.basename(output_path)}")
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", voice_path,
            "-acodec", "aac", "-b:a", "192k",
            output_path
        ]
        log(f"🎬 [ffmpeg-mixer] voice only → {os.path.basename(output_path)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio mix failed:\n{result.stderr[-500:]}")
    return output_path


def create_final_tiktok_video(input_vdo, voice_audio, output_path, title_text, srt_path, bg_audio_path=None, log_func=None):
    def log(m):
        if log_func: log_func(m)
        else: print(m)

    log("🎬 [Composer] Loading assets...")
    video = VideoFileClip(input_vdo).without_audio()

    # --- Compute bg volume factor (RMS-normalized) ---
    log(f"🔍 [Mixer] bg_audio_path = {bg_audio_path}")
    log(f"🔍 [Mixer] bg_audio exists = {os.path.exists(bg_audio_path) if bg_audio_path else False}")

    bg_vol_factor = 0.0
    if bg_audio_path and os.path.exists(bg_audio_path):
        log(f"🔍 [Mixer] bg_audio size = {os.path.getsize(bg_audio_path):,} bytes")
        voice_rms = measure_active_rms(voice_audio)
        bg_rms    = measure_rms(bg_audio_path)

        # Target bg at 40% of active voice loudness (audible but not overwhelming)
        BG_TARGET_RATIO = 0.40
        bg_vol_factor = (BG_TARGET_RATIO * voice_rms) / bg_rms if bg_rms > 0 else 0.30
        # Clamp: at least 0.3 (audible), at most 3.0 (not ear-splitting)
        bg_vol_factor = max(0.30, min(3.0, bg_vol_factor))
        log(f"🎬 [Mixer] voice active-RMS={voice_rms:.4f}  bg RMS={bg_rms:.4f}  → bg_vol={bg_vol_factor:.3f}")
    else:
        log("⚠️ [Composer] No background audio found, using voice only")

    # --- Mix audio with ffmpeg (reliable) ---
    ws_dir = os.path.dirname(voice_audio)
    mixed_audio_path = os.path.join(ws_dir, "mixed_audio.aac")
    mix_audio_with_ffmpeg(
        voice_audio,
        bg_audio_path if bg_vol_factor > 0 else None,
        mixed_audio_path,
        bg_vol_factor,
        log_func=log
    )

    voice_clip = AudioFileClip(voice_audio)
    render_duration = voice_clip.duration
    mixed_audio = AudioFileClip(mixed_audio_path)
    video = video.set_duration(render_duration).set_audio(mixed_audio)

    canvas_w, canvas_h = 1080, 1920
    
    # --- DYNAMIC TITLE BAR CALCULATION ---
    wrapped_title, num_lines = wrap_thai(title_text)
    if num_lines == 1:
        title_bar_h = 280
    else:
        title_bar_h = 420
    
    log(f"🎬 [Composer] Title has {num_lines} line(s), Setting bar height to {title_bar_h}")
    
    log("🎬 [SmartCrop] Analyzing best crop area...")
    area = detect_content_area(input_vdo)
    if area:
        min_x, min_y, max_x, max_y = area
        content_w = max_x - min_x
        content_h = max_y - min_y
        center_x = min_x + (content_w / 2)
        center_y = min_y + (content_h / 2)
        log(f"   - Detected content: {content_w}x{content_h} at center ({center_x}, {center_y})")
    else:
        center_x, center_y = video.w / 2, video.h / 2
        log("   - Could not detect content, using center crop.")

    # Resize to fill 1080 width at least
    video_sq = video.resize(width=canvas_w)
    if video_sq.h < 1080:
        video_sq = video.resize(height=1080)
    
    # Adjust center_x, center_y for resized video
    scale = video_sq.w / video.w
    scaled_center_x = center_x * scale
    scaled_center_y = center_y * scale

    log("🎬 [Composer] Applying 1:1 Crop...")
    video_sq = video_sq.crop(x_center=scaled_center_x, y_center=scaled_center_y, width=1080, height=1080)

    # วางวิดีโอชิดขอบบนสุด (ใต้ Title Bar)
    video_y_pos = title_bar_h

    log("🎬 [Composer] Adding UI elements (Bars/Title)...")
    bg_title = ColorClip(size=(canvas_w, title_bar_h), color=(255, 0, 80)).set_duration(video.duration)
    title_clip = TextClip(
        wrapped_title, fontsize=95, color='white', font=FONT_PATH,
        method='caption', size=(int(canvas_w * 0.9), title_bar_h), align='North'
    ).set_duration(video.duration).set_position(('center', 'top'))

    sub_bar_h, sub_y_pos = 130, title_bar_h + 1080
    blue_bar = ColorClip(size=(canvas_w, sub_bar_h), color=(0, 242, 234)).set_duration(video.duration).set_position((0, sub_y_pos))

    log("🎬 [Composer] Rendering subtitles...")
    temp_subs = SubtitlesClip(srt_path, lambda x: TextClip(x, font=FONT_PATH, fontsize=60))
    sub_clips = []
    for (st, et), txt in temp_subs.subtitles:
        txt = txt.strip()
        if len(txt) <= 38:
            c = TextClip(txt, fontsize=62, color='black', font=FONT_PATH).set_start(st).set_end(et).set_position(('center', sub_y_pos + 30))
            sub_clips.append(c)
        else:
            l1, l2 = split_thai_sub(txt)
            mid = st + ((et - st) / 2)
            sub_clips.append(TextClip(l1, fontsize=62, color='black', font=FONT_PATH).set_start(st).set_end(mid).set_position(('center', sub_y_pos + 30)))
            if l2.strip():
                sub_clips.append(TextClip(l2, fontsize=62, color='black', font=FONT_PATH).set_start(mid).set_end(et).set_position(('center', sub_y_pos + 30)))

    bg_black = ColorClip(size=(canvas_w, canvas_h), color=(0,0,0)).set_duration(video.duration)
    final = CompositeVideoClip([bg_black, bg_title, title_clip, video_sq.set_position((0, video_y_pos)), blue_bar] + sub_clips, size=(canvas_w, canvas_h))
    
    log(f"🎬 [Composer] Rendering final MP4: {output_path}...")
    final.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac", threads=4, logger=None)
    log("✅ Video composition complete.")

# --- 4. MAIN RUNNER ---

async def run_pipeline(video_url, playlist_name, use_voxcpm_tts=True, use_anime=True, progress_callback=None):
    def log(msg):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    try:
        global WS_DIR, OUTPUT_DIR, RAW_VIDEO, ANIME_VIDEO, THAI_SUB, THAI_VOICE, DATA_JSON, ENG_NAME, USE_VOXCPM_TTS, USE_ANIME
        
        USE_VOXCPM_TTS = use_voxcpm_tts
        USE_ANIME = use_anime
        OUTPUT_DIR = f"outputs/{playlist_name}"

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Resolve shortened URL
        log(f"🔍 Checking URL: {video_url}")
        video_url = resolve_tiktok_url(video_url)
        
        video_id = video_url.split('/')[-1].split('?')[0]
        if not video_id or not video_id.isdigit():
            # Fallback for non-standard full URLs or if resolution didn't result in a numeric ID
            video_id = video_url.split('/')[-1].split('?')[0]
            
        ENG_NAME = f"tiktok_{video_id}"

        log(f"🔥 Starting Pipeline for: {ENG_NAME} (Full URL: {video_url})")

        job_ws_dir = os.path.join(WS_DIR, ENG_NAME)
        os.makedirs(job_ws_dir, exist_ok=True)

        raw_video_path = os.path.join(job_ws_dir, "input_video.mp4")
        anime_video_path = os.path.join(job_ws_dir, "input_video_anime.mp4")
        thai_sub_path = os.path.join(job_ws_dir, "thai_sub.srt")
        thai_voice_path = os.path.join(job_ws_dir, "thai_dub.mp3")
        data_json_path = os.path.join(job_ws_dir, "data.json")
        bg_audio_path = os.path.join(job_ws_dir, "separated/htdemucs/input_video/no_vocals.wav")

        # Step 1: Download
        if is_valid_video(raw_video_path):
            log("⏭️ Step 1: Video already downloaded & valid, skipping...")
        else:
            log("🚀 Step 1: Downloading source video...")
            r = await asyncio.to_thread(download_video_to_path, video_url, raw_video_path)
            if r['status'] == 'error':
                raise Exception(f"Download failed: {r['message']}")

        # Step 2: Transcribe & Translate
        if is_valid_srt(thai_sub_path):
            log("⏭️ Step 2: Thai subtitles already valid, skipping...")
        else:
            log("🚀 Step 2: Transcribing & Translating...")
            await asyncio.to_thread(generate_thai_sub_to_path, raw_video_path, thai_sub_path, job_ws_dir, log_func=log)

        # Step 3: Separate Background Audio
        if is_valid_audio(bg_audio_path):
            log("⏭️ Step 3: Background audio already valid, skipping...")
        else:
            log("🚀 Step 3: Separating Background Audio...")
            await asyncio.to_thread(separate_bg_audio_to_path, raw_video_path, job_ws_dir)

        # Step 4: Generate Thai Dubbing
        if is_valid_audio(thai_voice_path):
            log("⏭️ Step 4: Thai dubbing already valid, skipping...")
        else:
            log("🚀 Step 4: Generating Thai Dubbing...")
            await make_final_audio_to_path(thai_sub_path, thai_voice_path, job_ws_dir, use_voxcpm_tts, log_func=log)

        # Step 5: Anime (optional)
        if use_anime:
            if is_valid_video(anime_video_path):
                log("⏭️ Step 5: Anime video already valid, skipping...")
            else:
                log("🎨 Step 5: Applying Anime Style Conversion...")
                await asyncio.to_thread(run_anime_conversion_to_path, raw_video_path, anime_video_path, log_func=log)
            final_input_vdo = anime_video_path
            suffix = "_anime"
        else:
            final_input_vdo = raw_video_path
            suffix = ""

        if use_voxcpm_tts:
            suffix += "_voxcpm"

        # Step 6: AI Suggestions
        if is_valid_json(data_json_path):
            log("⏭️ Step 6: AI suggestions already valid, skipping...")
            with open(data_json_path, 'r', encoding='utf-8') as f:
                ai_data = json.load(f)
        else:
            log("🤖 Step 6: Getting AI Content Suggestions (Ollama/Llama3)...")
            ai_data = {}
            for i in range(5):
                ai_data = await asyncio.to_thread(get_ollama_suggestions_to_path, thai_sub_path, data_json_path, log_func=log)
                is_valid = all(v is not None for v in ai_data.values())
                if is_valid: break

        title_hook = ai_data.get('title_hook', 'สรุปคลิปที่น่าสนใจ')

        final_vdo_path = os.path.join(OUTPUT_DIR, f"tk_{ENG_NAME}{suffix}.mp4")
        final_json_path = os.path.join(OUTPUT_DIR, f"tk_{ENG_NAME}.json")

        # Step 7: Final Composition (always re-render)
        log("🎬 Step 7: Compositing Final TikTok Layout...")
        await asyncio.to_thread(create_final_tiktok_video, final_input_vdo, thai_voice_path, final_vdo_path, title_hook, thai_sub_path, bg_audio_path=bg_audio_path, log_func=log)

        if os.path.exists(data_json_path):
            shutil.copy(data_json_path, final_json_path)

        log(f"✅ Pipeline Finished! Output saved to: {final_vdo_path}")

        # Cleanup workspace only after final output is confirmed
        if os.path.exists(final_vdo_path):
            log(f"🧹 Cleaning up workspace: {job_ws_dir}")
            shutil.rmtree(job_ws_dir)

        return {"status": "success", "video": os.path.abspath(final_vdo_path), "json": os.path.abspath(final_json_path)}

    except Exception as e:
        log(f"❌ Critical Pipeline Error: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # For backward compatibility or CLI testing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=VIDEO_URL)
    parser.add_argument("--playlist", default=PLAYLIST_NAME)
    parser.add_argument("--voxcpm", action="store_true", default=USE_VOXCPM_TTS)
    parser.add_argument("--anime", action="store_true", default=USE_ANIME)
    args = parser.parse_args()
    
    asyncio.run(run_pipeline(args.url, args.playlist, args.voxcpm, args.anime))