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
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from moviepy.editor import (VideoFileClip, AudioFileClip, CompositeAudioClip,
                            TextClip, CompositeVideoClip, ColorClip, ImageClip)
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
FONT_PATH_LATIN = "/Library/Fonts/Arial Unicode.ttf"  # supports Finnish/Latin Extended diacritics
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

def _ffprobe_duration(video_path):
    """Return video duration in seconds via ffprobe, or 0.0 on error."""
    import subprocess
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True
    )
    try:
        return float(probe.stdout.strip())
    except Exception:
        return 0.0


def _find_first_bright_frame(video_path, brightness_threshold=25):
    """Return the first frame (as numpy RGB array) whose mean brightness exceeds the threshold.
    Falls back to frame 0 if no bright frame is found within the first 10 seconds."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    max_frames = int(fps * 10)  # search up to 10s
    first_frame = None
    for _ in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if first_frame is None:
            first_frame = frame
        if cv2.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[0] >= brightness_threshold:
            cap.release()
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    if first_frame is not None:
        return cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    return None


def _next_job_number(output_dir):
    """Return the next sequential job number for output_dir, persisted in .counter file."""
    counter_file = os.path.join(output_dir, ".counter")
    try:
        with open(counter_file, 'r') as f:
            n = int(f.read().strip())
    except Exception:
        n = 0
    n += 1
    with open(counter_file, 'w') as f:
        f.write(str(n))
    return n


def _find_sentence_split_points(srt_path, max_length_sec, total_dur_sec):
    """Find split points at subtitle/sentence boundaries so each segment ≤ max_length_sec.
    Returns list of (start_sec, end_sec) tuples — lengths vary but never exceed max_length_sec."""
    subs = pysubs2.load(srt_path)
    segments = []
    seg_start = 0.0
    last_good_end = 0.0  # end time of last subtitle that still fits

    for sub in subs:
        sub_end_sec = sub.end / 1000.0
        if sub_end_sec - seg_start <= max_length_sec:
            # This subtitle still fits in the current segment
            last_good_end = sub_end_sec
        else:
            # Adding this subtitle would exceed max_length
            if last_good_end > seg_start:
                # Cut after the last subtitle that fitted
                segments.append((seg_start, last_good_end))
                seg_start = last_good_end
                # Now check if current subtitle fits from new seg_start
                if sub_end_sec - seg_start <= max_length_sec:
                    last_good_end = sub_end_sec
                else:
                    # Single subtitle is longer than max_length — cut at its start
                    sub_start_sec = sub.start / 1000.0
                    if sub_start_sec > seg_start:
                        segments.append((seg_start, sub_start_sec))
                        seg_start = sub_start_sec
                    last_good_end = sub_end_sec
            else:
                # No good cut point at all — force cut at this subtitle's start
                sub_start_sec = sub.start / 1000.0
                if sub_start_sec > seg_start:
                    segments.append((seg_start, sub_start_sec))
                    seg_start = sub_start_sec
                last_good_end = sub_end_sec

    # Flush last segment
    if seg_start < total_dur_sec:
        segments.append((seg_start, total_dur_sec))

    return segments


def _extract_srt_segment(subs, start_sec, end_sec, output_path):
    """Extract subtitle events within [start_sec, end_sec] and shift timestamps to start from 0."""
    start_ms = int(start_sec * 1000)
    end_ms   = int(end_sec   * 1000)
    out = pysubs2.SSAFile()
    for sub in subs:
        if sub.end <= start_ms or sub.start >= end_ms:
            continue
        new_sub = sub.copy()
        new_sub.start = max(0, sub.start - start_ms)
        new_sub.end   = min(end_ms - start_ms, sub.end - start_ms)
        out.append(new_sub)
    out.save(output_path)


def unique_path(path):
    """Return path unchanged if it doesn't exist, otherwise append _001, _002, …"""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        candidate = f"{base}_{i:03d}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1

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

def generate_thai_sub_to_path(video_path, srt_path, ws_dir, log_func=None, eng_srt_path=None,
                              input_lang='en', output_lang='th'):
    """Transcribe video with Whisper (input_lang) and optionally translate to output_lang.
    eng_srt_path receives the raw transcription (source language).
    srt_path receives the output language SRT (translated, or same as source if langs match)."""
    def log(m):
        if log_func: log_func(m)
        else: print(m)

    log(f"🚀 [Whisper] Loading model (base)... input={input_lang} output={output_lang}")
    model = whisper.load_model("base")
    log(f"🚀 [Whisper] Transcribing ({input_lang})...")
    result = model.transcribe(video_path, language=input_lang)

    # Save source-language SRT before translation
    if eng_srt_path:
        src_subs = pysubs2.SSAFile()
        for seg in result['segments']:
            src_subs.append(pysubs2.SSAEvent(
                start=int(seg['start'] * 1000),
                end=int(seg['end'] * 1000),
                text=seg['text'].strip()
            ))
        src_subs.save(eng_srt_path)
        log(f"✅ Source SRT saved: {eng_srt_path}")

    # Translate only if languages differ
    if input_lang != output_lang:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source=input_lang, target=output_lang)
        log(f"🚀 [Translator] Translating {len(result['segments'])} segments ({input_lang}→{output_lang})...")
        for i, segment in enumerate(result['segments']):
            try:
                segment['text'] = translator.translate(segment['text'])
                if i % 5 == 0: log(f"   - Translated {i+1}/{len(result['segments'])}...")
            except Exception as e:
                log(f"   - Translation error at segment {i}: {e}")
    else:
        log(f"⏭️ [Translator] Same language ({input_lang}), skipping translation.")

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

def _write_synced_srt(timing_data, synced_srt_path):
    """Write an SRT file from a list of {start, end, text} dicts (times in seconds)."""
    synced = pysubs2.SSAFile()
    for entry in timing_data:
        e = pysubs2.SSAEvent(
            start=int(entry["start"] * 1000),
            end=int(entry["end"] * 1000),
            text=entry["text"]
        )
        synced.append(e)
    synced.save(synced_srt_path)


async def make_final_audio_to_path(srt_path, output_audio, ws_dir, use_voxcpm, log_func=None, synced_srt_path=None):
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

        clip_duration = audio_clip.duration
        # Record actual timing for synced SRT
        line._actual_start = actual_start
        line._actual_end = actual_start + clip_duration

        # Update last_end_time for the next clip
        last_end_time = actual_start + clip_duration

    log("🚀 [Mixer] Combining Thai voice dubs...")
    final_mix = CompositeAudioClip(dub_clips)
    final_mix.write_audiofile(output_audio, fps=44100, logger=None)
    log(f"✅ Thai voice audio complete. Total duration: {final_mix.duration:.2f}s")

    # Save timing JSON (so synced SRT can be rebuilt later without re-running TTS)
    timing_json_path = output_audio.replace(".mp3", "_timing.json")
    timing_data = []
    for line in subs:
        if not line.text.strip() or not hasattr(line, '_actual_start'):
            continue
        timing_data.append({"start": line._actual_start, "end": line._actual_end, "text": line.text})
    with open(timing_json_path, "w", encoding="utf-8") as f:
        json.dump(timing_data, f, ensure_ascii=False, indent=2)

    # Write synced SRT with actual clip timing
    if synced_srt_path:
        _write_synced_srt(timing_data, synced_srt_path)
        log(f"✅ Synced SRT saved: {synced_srt_path}")

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
    # Clamp mixed_audio to render_duration: AAC encoder delay can make the file
    # slightly shorter than voice, causing MoviePy to loop audio at the end.
    mixed_audio = AudioFileClip(mixed_audio_path).set_duration(render_duration)
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


def create_method2_video(input_vdo, voice_audio, output_path, srt_path=None, bg_audio_path=None, log_func=None):
    """Method 2: content video (top) + red subtitle bar (middle) + looped avatar video (bottom)."""
    import random, glob
    def log(m):
        if log_func: log_func(m)
        else: print(m)

    AVATARS_DIR = "avartars_no_sound"
    canvas_w, canvas_h = 1080, 1920
    sub_bar_h = 130
    content_h = (canvas_h - sub_bar_h) // 2  # 895
    avatar_h = canvas_h - sub_bar_h - content_h  # 895
    sub_y_pos = content_h  # 895
    avatar_y_pos = content_h + sub_bar_h  # 1025

    avatar_files = glob.glob(os.path.join(AVATARS_DIR, "*.mp4"))
    if not avatar_files:
        raise FileNotFoundError(f"No avatar videos found in {AVATARS_DIR}/")
    avatar_path = random.choice(avatar_files)
    log(f"🎲 [Method2] Using avatar: {os.path.basename(avatar_path)}")

    ws_dir = os.path.dirname(voice_audio)
    render_duration, mixed_audio = _prepare_avatar_audio(voice_audio, bg_audio_path, ws_dir, "m2", log)

    log("🎬 [Method2] Loading content video...")
    content_raw = VideoFileClip(input_vdo).without_audio().set_duration(render_duration)
    # Fit content into 1080 x content_h (top section)
    if content_raw.w / content_raw.h < canvas_w / content_h:
        content = content_raw.resize(width=canvas_w)
    else:
        content = content_raw.resize(height=content_h)
    content = content.crop(x_center=content.w / 2, y_center=content.h / 2, width=canvas_w, height=content_h)
    content = content.set_duration(render_duration)

    log("🎬 [Method2] Loading & looping avatar video...")
    avatar = _pick_and_loop_avatar(avatar_path, render_duration, canvas_w, avatar_h)

    log("🎬 [Method2] Building subtitle bar...")
    red_bar = ColorClip(size=(canvas_w, sub_bar_h), color=(220, 0, 0)).set_duration(render_duration).set_position((0, sub_y_pos))

    sub_clips = []
    if srt_path and os.path.exists(srt_path):
        temp_subs = SubtitlesClip(srt_path, lambda x: TextClip(x, font=FONT_PATH, fontsize=60))
        for (st, et), txt in temp_subs.subtitles:
            txt = txt.strip()
            if len(txt) <= 38:
                c = TextClip(txt, fontsize=62, color='white', font=FONT_PATH).set_start(st).set_end(et).set_position(('center', sub_y_pos + 30))
                sub_clips.append(c)
            else:
                l1, l2 = split_thai_sub(txt)
                mid = st + ((et - st) / 2)
                sub_clips.append(TextClip(l1, fontsize=62, color='white', font=FONT_PATH).set_start(st).set_end(mid).set_position(('center', sub_y_pos + 30)))
                if l2.strip():
                    sub_clips.append(TextClip(l2, fontsize=62, color='white', font=FONT_PATH).set_start(mid).set_end(et).set_position(('center', sub_y_pos + 30)))

    log("🎬 [Method2] Compositing vertical stack...")
    bg_black = ColorClip(size=(canvas_w, canvas_h), color=(0, 0, 0)).set_duration(render_duration)
    final = CompositeVideoClip(
        [bg_black, content.set_position((0, 0)), red_bar, avatar.set_position((0, avatar_y_pos))] + sub_clips,
        size=(canvas_w, canvas_h)
    ).set_audio(mixed_audio)

    log(f"🎬 [Method2] Rendering final MP4: {output_path}...")
    final.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac", threads=4, logger=None)
    log("✅ Method2 composition complete.")


def _prepare_avatar_audio(voice_audio, bg_audio_path, ws_dir, suffix, log):
    """Shared helper: compute bg_vol, mix audio with ffmpeg, return (render_duration, mixed_audio)."""
    bg_vol_factor = 0.0
    log(f"🔍 [Mixer] bg_audio_path = {bg_audio_path}")
    log(f"🔍 [Mixer] bg_audio exists = {os.path.exists(bg_audio_path) if bg_audio_path else False}")
    if bg_audio_path and os.path.exists(bg_audio_path):
        log(f"🔍 [Mixer] bg_audio size = {os.path.getsize(bg_audio_path):,} bytes")
        voice_rms = measure_active_rms(voice_audio)
        bg_rms    = measure_rms(bg_audio_path)
        bg_vol_factor = (0.40 * voice_rms) / bg_rms if bg_rms > 0 else 0.30
        bg_vol_factor = max(0.30, min(3.0, bg_vol_factor))
        log(f"🎬 [Mixer] voice active-RMS={voice_rms:.4f}  bg RMS={bg_rms:.4f}  → bg_vol={bg_vol_factor:.3f}")
    else:
        log(f"⚠️ No background audio, using voice only")

    mixed_audio_path = os.path.join(ws_dir, f"mixed_audio_{suffix}.aac")
    mix_audio_with_ffmpeg(
        voice_audio,
        bg_audio_path if bg_vol_factor > 0 else None,
        mixed_audio_path,
        bg_vol_factor,
        log_func=log
    )
    voice_clip      = AudioFileClip(voice_audio)
    render_duration = voice_clip.duration
    mixed_audio     = AudioFileClip(mixed_audio_path)
    # Clamp render_duration to actual mixed audio length to avoid read-past-end errors
    # (AAC encoding can produce a file slightly shorter than the source voice)
    render_duration = min(render_duration, mixed_audio.duration)
    mixed_audio     = mixed_audio.set_duration(render_duration)
    return render_duration, mixed_audio


def _pick_and_loop_avatar(avatar_path, render_duration, target_w, target_h):
    """Resize and loop an avatar clip to exactly target_w x target_h x render_duration."""
    raw = VideoFileClip(avatar_path).without_audio()
    looped = raw.loop(duration=render_duration)
    # Fill target width first, then crop height (or vice versa)
    if looped.w / looped.h < target_w / target_h:
        looped = looped.resize(width=target_w)
    else:
        looped = looped.resize(height=target_h)
    looped = looped.crop(x_center=looped.w / 2, y_center=looped.h / 2, width=target_w, height=target_h)
    return looped.set_duration(render_duration)


def create_method3_video(input_vdo, voice_audio, output_path, srt_path=None, bg_audio_path=None, log_func=None):
    """Method 3: YouTube 16:9 — content video (left half) + looped avatar (right half) + red subtitle bar at bottom."""
    import random, glob
    def log(m):
        if log_func: log_func(m)
        else: print(m)

    AVATARS_DIR = "avartars_no_sound"
    canvas_w, canvas_h = 1920, 1080  # YouTube 16:9
    half_w = canvas_w // 2  # 960
    sub_bar_h = 120
    video_h = canvas_h - sub_bar_h  # 960
    sub_y_pos = video_h  # 960

    avatar_files = glob.glob(os.path.join(AVATARS_DIR, "*.mp4"))
    if not avatar_files:
        raise FileNotFoundError(f"No avatar videos found in {AVATARS_DIR}/")
    avatar_path = random.choice(avatar_files)
    log(f"🎲 [Method3] Using avatar: {os.path.basename(avatar_path)}")

    ws_dir = os.path.dirname(voice_audio)
    render_duration, mixed_audio = _prepare_avatar_audio(voice_audio, bg_audio_path, ws_dir, "m3", log)

    log("🎬 [Method3] Loading content video...")
    content_raw = VideoFileClip(input_vdo).without_audio().set_duration(render_duration)
    # Fit content into 540 x 1790 (left half, above subtitle bar)
    if content_raw.w / content_raw.h < half_w / video_h:
        content = content_raw.resize(width=half_w)
    else:
        content = content_raw.resize(height=video_h)
    content = content.crop(x_center=content.w / 2, y_center=content.h / 2, width=half_w, height=video_h)
    content = content.set_duration(render_duration)

    log("🎬 [Method3] Loading & looping avatar video...")
    avatar = _pick_and_loop_avatar(avatar_path, render_duration, half_w, video_h)

    log("🎬 [Method3] Building subtitle bar...")
    red_bar = ColorClip(size=(canvas_w, sub_bar_h), color=(220, 0, 0)).set_duration(render_duration).set_position((0, sub_y_pos))

    sub_clips = []
    if srt_path and os.path.exists(srt_path):
        temp_subs = SubtitlesClip(srt_path, lambda x: TextClip(x, font=FONT_PATH, fontsize=60))
        text_y = sub_y_pos + (sub_bar_h - 62) // 2  # vertically center 62px text in bar
        for (st, et), txt in temp_subs.subtitles:
            txt = txt.strip()
            if len(txt) <= 38:
                c = TextClip(txt, fontsize=62, color='white', font=FONT_PATH).set_start(st).set_end(et).set_position(('center', text_y))
                sub_clips.append(c)
            else:
                l1, l2 = split_thai_sub(txt)
                mid = st + ((et - st) / 2)
                sub_clips.append(TextClip(l1, fontsize=62, color='white', font=FONT_PATH).set_start(st).set_end(mid).set_position(('center', text_y)))
                if l2.strip():
                    sub_clips.append(TextClip(l2, fontsize=62, color='white', font=FONT_PATH).set_start(mid).set_end(et).set_position(('center', text_y)))

    log("🎬 [Method3] Compositing horizontal split...")
    bg_black = ColorClip(size=(canvas_w, canvas_h), color=(0, 0, 0)).set_duration(render_duration)
    final = CompositeVideoClip(
        [
            bg_black,
            content.set_position((0, 0)),
            avatar.set_position((half_w, 0)),
            red_bar,
        ] + sub_clips,
        size=(canvas_w, canvas_h)
    ).set_audio(mixed_audio)

    log(f"🎬 [Method3] Rendering final MP4: {output_path}...")
    final.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac", threads=4, logger=None)
    log("✅ Method3 composition complete.")


def _font_for_lang(lang, size):
    """Return a PIL ImageFont appropriate for the given language."""
    path = FONT_PATH if lang == 'th' else FONT_PATH_LATIN
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _wrap_for_lang(lang, text, font, draw, max_w, max_lines=2):
    """Choose word-wrap strategy based on language."""
    if lang == 'th':
        return _wrap_thai(text, font, draw, max_w, max_lines=max_lines)
    return _wrap_eng(text, font, draw, max_w, max_lines=max_lines)


def _safe_filename(text, max_len=60):
    """Convert arbitrary text into a safe filename slug."""
    import re as _re
    slug = _re.sub(r'[^\w\s-]', '', text).strip()
    slug = _re.sub(r'[\s]+', '_', slug)
    return slug[:max_len]


def generate_title_and_hashtags(srt_path, input_lang='en', output_lang='th', log_func=None):
    """Generate a short English title and hashtags from subtitle content via Ollama."""
    def log(m):
        if log_func: log_func(m)
        else: print(m)

    LANG_NAMES = {'en': 'English', 'fi': 'Finnish', 'th': 'Thai'}
    in_name  = LANG_NAMES.get(input_lang,  input_lang)
    out_name = LANG_NAMES.get(output_lang, output_lang)
    mandatory = [f"#learning{input_lang}", f"#learning{output_lang}"]

    # Read transcript text from SRT
    try:
        subs = pysubs2.load(srt_path)
        text = ' '.join(s.text.strip().replace('\\N', ' ') for s in subs[:40])
    except Exception:
        text = ''

    default = {
        "title": f"Learn {out_name} from {in_name}",
        "hashtags": mandatory + ["#vocabulary", "#language", "#learn"]
    }
    if not text:
        return default

    prompt = (
        f"Read this {in_name} transcript and generate:\n"
        f"1) A short catchy English title (4-7 words, no quotes, no colon)\n"
        f"2) 6-8 relevant hashtags. These two are MANDATORY: {' '.join(mandatory)}\n\n"
        f'Return ONLY valid JSON (no markdown): {{"title": "...", "hashtags": ["...", "..."]}}\n\n'
        f"Transcript:\n{text[:800]}"
    )

    try:
        import ollama, re as _re
        response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        raw = response['message']['content']
        m = _re.search(r'\{.*\}', raw, _re.DOTALL)
        if m:
            data = json.loads(m.group())
            title    = data.get('title', '').strip() or default['title']
            hashtags = data.get('hashtags', [])
            for tag in reversed(mandatory):
                if tag not in hashtags:
                    hashtags.insert(0, tag)
            log(f"✅ [Title] \"{title}\"  tags: {' '.join(hashtags)}")
            return {"title": title, "hashtags": hashtags}
    except Exception as e:
        log(f"⚠️ Title/hashtag generation failed: {e}")

    return default


def extract_vocab_with_ollama(src_srt_path, input_lang='en', output_lang='th',
                              clip_duration_sec=0, log_func=None):
    """Extract vocabulary words that MUST appear in the episode subtitle.
    Count depends on clip duration: <2 min → 6, >=2 min → 8, >=5 min → 12.
    Each word is validated against the actual SRT text. Retries if too few words pass."""
    import re as _re
    LANG_NAMES = {'en': 'English', 'fi': 'Finnish', 'th': 'Thai'}
    def log(m):
        if log_func: log_func(m)
        else: print(m)

    in_name  = LANG_NAMES.get(input_lang, input_lang)
    out_name = LANG_NAMES.get(output_lang, output_lang)

    # Dynamic vocab count based on clip duration
    if clip_duration_sec >= 300:   # >= 5 min
        min_words, max_words = 10, 12
    elif clip_duration_sec >= 120: # >= 2 min
        min_words, max_words = 8, 10
    else:                          # < 2 min
        min_words, max_words = 6, 8

    log(f"🤖 [Vocab-M4] Extracting {in_name} vocab → {out_name} translation... "
        f"(target {min_words}–{max_words} words, clip={clip_duration_sec:.0f}s)")

    if not os.path.exists(src_srt_path):
        log("⚠️ [Vocab-M4] Source SRT not found, returning empty list")
        return []

    # Build plain text from SRT (strip indices and timestamps)
    with open(src_srt_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
    src_text = " ".join(
        l.strip() for l in raw_lines
        if l.strip() and not l.strip().isdigit() and '-->' not in l
    )
    # Normalised text for validation (lowercase, letters only)
    text_words_norm = set(_re.findall(r"[a-zA-Z\u00C0-\u024F\u0E00-\u0E7F']+", src_text.lower()))

    def _word_in_text(word):
        """True if the word (or its base form) appears in the episode text."""
        w = word.lower().strip()
        # Direct presence check (case-insensitive substring)
        if w in src_text.lower():
            return True
        # Token-level match (handles inflections like "running"→"run" won't match,
        # but at least "runs" will match "runs")
        return w in text_words_norm

    def _normalise(entry):
        """Normalise Ollama response: unify 'translation'/'thai' key."""
        if 'translation' in entry:
            val = entry.pop('translation')
            if not entry.get('thai'):
                entry['thai'] = val
        return entry

    url = "http://localhost:11434/api/generate"

    def _ask_ollama(extra_instruction=""):
        prompt = (
            f"You are a language teacher. Choose {min_words}-{max_words} difficult or advanced {in_name} vocabulary words "
            f"FROM THE TEXT BELOW that would be worth teaching to learners. "
            f"IMPORTANT: every word you choose MUST appear verbatim in the text (exact spelling, any case). "
            f"Avoid very common/basic words. {extra_instruction}\n\n"
            f"Text:\n{src_text[:2000]}\n\n"
            f"Reply ONLY with valid JSON:\n"
            f'{{"vocab": [{{"word": "exact_word_from_text", "pos": "n", "translation": "{out_name} meaning"}}]}}\n'
            "POS: n=noun v=verb adj=adjective adv=adverb\n"
            f"'translation' must be in {out_name}."
        )
        payload = {"model": "llama3", "prompt": prompt, "format": "json", "stream": False}
        response = requests.post(url, json=payload, timeout=90)
        data = json.loads(response.json()['response'])
        raw = [_normalise(e) for e in data.get('vocab', []) if isinstance(e, dict) and e.get('word')]
        return raw

    vocab = []
    for attempt in range(3):
        try:
            extra = "Only pick words that are literally spelled out in the text." if attempt > 0 else ""
            candidates = _ask_ollama(extra)
        except Exception as e:
            log(f"⚠️ [Vocab-M4] Ollama attempt {attempt+1} failed: {e}")
            candidates = []

        # Validate: keep only words actually present in this episode's text
        valid = [e for e in candidates if _word_in_text(e.get('word', ''))]
        rejected = [e['word'] for e in candidates if not _word_in_text(e.get('word', ''))]
        if rejected:
            log(f"   ⚠️ Removed {len(rejected)} hallucinated words: {', '.join(rejected)}")

        # Merge new valid words with previously found ones (avoid duplicates)
        seen = {e['word'].lower() for e in vocab}
        for e in valid:
            if e['word'].lower() not in seen:
                vocab.append(e)
                seen.add(e['word'].lower())

        if len(vocab) >= min_words:
            break
        log(f"   🔄 Only {len(vocab)} valid words so far, retrying ({attempt+1}/3)...")

    vocab = vocab[:max_words]
    log(f"✅ [Vocab-M4] {len(vocab)} validated words ({in_name}→{out_name})")

    # Fill any missing translations via deep_translator
    if output_lang != input_lang:
        try:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source=input_lang, target=output_lang)
            for entry in vocab:
                if not entry.get('thai', '').strip():
                    try:
                        entry['thai'] = translator.translate(entry['word'])
                        log(f"   - Translated: {entry['word']} → {entry['thai']}")
                    except Exception:
                        pass
        except Exception as e:
            log(f"⚠️ [Vocab-M4] Translation fallback failed: {e}")

    return vocab


def _wrap_eng(text, font, draw, max_w, max_lines=2):
    """Word-wrap English text up to max_lines lines."""
    words = text.split()
    lines = []
    current = []
    for word in words:
        test = ' '.join(current + [word])
        bb = draw.textbbox((0, 0), test, font=font)
        if bb[2] - bb[0] > max_w:
            if current:
                lines.append(' '.join(current))
                current = [word]
            else:
                lines.append(word)  # single word wider than max_w — accept as-is
                current = []
        else:
            current.append(word)
    if current:
        lines.append(' '.join(current))
    return lines[:max_lines]


def _wrap_thai(text, font, draw, max_w, max_lines=2):
    """Wrap Thai text at word boundaries (pythainlp). Max 2 lines per zone."""
    words = word_tokenize(text, engine="newmm")
    lines, current = [], ""
    for word in words:
        test = current + word
        bb = draw.textbbox((0, 0), test, font=font)
        if current and bb[2] - bb[0] > max_w:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    return [l for l in lines[:max_lines] if l]


def _render_m4_subtitle_bar(src_lines, dst_lines, highlight_words, width, slot_h=90,
                             src_lang='en', dst_lang='th', font_size_src=52, font_size_dst=50):
    """Render bilingual subtitle bar.
    Zone heights are based on actual line counts.
    Top zone: src_lang lines. Bottom zone: dst_lang lines (if any)."""
    import re

    n_src = max(1, len(src_lines))
    n_dst = max(1, len(dst_lines)) if dst_lines else 0
    SRC_H = n_src * slot_h
    DST_H = n_dst * slot_h
    height = SRC_H + DST_H

    img  = Image.new('RGB', (width, height), color=(45, 45, 45))
    draw = ImageDraw.Draw(img)

    if dst_lines:
        draw.line([(40, SRC_H), (width - 40, SRC_H)], fill=(75, 75, 75), width=1)

    font_src = _font_for_lang(src_lang, font_size_src)
    font_dst = _font_for_lang(dst_lang, font_size_dst)

    highlight_set = {re.sub(r"[^a-zA-Z\u00C0-\u024F']", '', w).lower() for w in highlight_words}

    def _center_x(text, font):
        bb = draw.textbbox((0, 0), text, font=font)
        return max(20, (width - (bb[2] - bb[0])) // 2) - bb[0]

    def zone_line_y(zone_start, zone_h, n_lines, line_idx, font):
        bb = draw.textbbox((0, 0), "Ag", font=font)
        line_h = bb[3] - bb[1]
        total_h = n_lines * line_h + max(0, n_lines - 1) * 8
        offset = (zone_h - total_h) // 2
        return zone_start + offset + line_idx * (line_h + 8)

    def draw_src_line(line_idx, line):
        words = line.split()
        if not words:
            return
        full = ' '.join(words)
        x0 = _center_x(full, font_src)
        y  = zone_line_y(0, SRC_H, len(src_lines), line_idx, font_src)
        draw.text((x0, y), full, fill=(255, 255, 255), font=font_src)
        # Highlight vocab words in red
        for i, word in enumerate(words):
            clean = re.sub(r"[^a-zA-Z\u00C0-\u024F']", '', word).lower()
            if clean not in highlight_set:
                continue
            if i == 0:
                wx = x0
            else:
                pb = draw.textbbox((0, 0), ' '.join(words[:i]) + ' ', font=font_src)
                wx = x0 + (pb[2] - pb[0])
            draw.text((wx, y), word, fill=(255, 80, 80), font=font_src)

    def draw_dst_line(line_idx, line):
        if not line.strip():
            return
        x = _center_x(line, font_dst)
        y = zone_line_y(SRC_H, DST_H, len(dst_lines), line_idx, font_dst)
        draw.text((x, y), line, fill=(180, 220, 255), font=font_dst)

    for i, line in enumerate(src_lines[:2]):
        draw_src_line(i, line)
    for i, line in enumerate(dst_lines[:2]):
        draw_dst_line(i, line)

    return img


def _wrap_text_to_lines(text, font, draw, max_w, lang='en'):
    """Wrap text into as many lines as needed to fit max_w, respecting word boundaries."""
    if lang == 'th':
        from pythainlp.tokenize import word_tokenize as _wt
        words = _wt(text, engine="newmm")
        lines, current = [], ""
        for word in words:
            test = current + word
            bb = draw.textbbox((0, 0), test, font=font)
            if current and bb[2] - bb[0] > max_w:
                lines.append(current)
                current = word
            else:
                current = test
        if current:
            lines.append(current)
    else:
        words = text.split()
        lines, current = [], []
        for word in words:
            test = ' '.join(current + [word])
            bb = draw.textbbox((0, 0), test, font=font)
            if bb[2] - bb[0] > max_w:
                if current:
                    lines.append(' '.join(current))
                    current = [word]
                else:
                    lines.append(word)
                    current = []
            else:
                current.append(word)
        if current:
            lines.append(' '.join(current))
    return lines or [text]


def _create_m4_vocab_image(vocab_list, width, height, src_lang='en', dst_lang='th', top_padding_lines=0):
    """Create a notebook-style PIL image with vocabulary list.
    Words are in src_lang, translations in dst_lang — fonts chosen accordingly.
    top_padding_lines: blank lines to skip before the header (e.g. 1 for YouTube layout).
    Long translations wrap to subsequent lines."""
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    line_spacing = 58
    for y in range(0, height, line_spacing):
        draw.line([(0, y), (width, y)], fill=(190, 220, 250), width=1)
    margin_x = 50
    draw.line([(margin_x, 0), (margin_x, height)], fill=(255, 170, 170), width=2)

    # Header: Latin font (always); word: src font; translation: dst font
    font_header = _font_for_lang('en', 48)   # "Vocabulary" header always Latin
    font_word   = _font_for_lang(src_lang, 38)
    font_trans  = _font_for_lang(dst_lang, 38)

    x_start = margin_x + 16
    max_text_w = width - x_start - 10  # usable text width
    indent_x = x_start + 24            # indent for wrapped translation lines

    y_pos = 5 + top_padding_lines * line_spacing

    draw.text((x_start, y_pos), "Vocabulary", fill=(0, 110, 220), font=font_header)
    y_pos += line_spacing

    for entry in vocab_list:
        word  = entry.get('word', '')
        pos   = entry.get('pos', 'n')
        trans = entry.get('thai', '')  # holds the translation regardless of language
        if y_pos + line_spacing > height:
            break

        # Line 1: "word (pos) — translation" if it fits; else translation goes to next line
        word_part = f"{word} ({pos})  "
        bb_word = draw.textbbox((0, 0), word_part, font=font_word)
        word_part_w = bb_word[2] - bb_word[0]

        trans_lines = _wrap_text_to_lines(trans, font_trans, draw, max_text_w - word_part_w, lang=dst_lang)
        first_fits = True
        if trans_lines:
            bb_first = draw.textbbox((0, 0), trans_lines[0], font=font_trans)
            if word_part_w + (bb_first[2] - bb_first[0]) > max_text_w:
                first_fits = False

        if first_fits and len(trans_lines) <= 1:
            # Everything fits on one line
            draw.text((x_start, y_pos), word_part, fill=(30, 30, 30), font=font_word)
            draw.text((x_start + word_part_w, y_pos), trans, fill=(0, 100, 200), font=font_trans)
            y_pos += line_spacing
        else:
            # Word part on its own line, then translation lines indented
            draw.text((x_start, y_pos), word_part, fill=(30, 30, 30), font=font_word)
            y_pos += line_spacing
            # Re-wrap translation to full width minus indent
            trans_lines = _wrap_text_to_lines(trans, font_trans, draw, max_text_w - (indent_x - x_start), lang=dst_lang)
            for tline in trans_lines:
                if y_pos + line_spacing > height:
                    break
                draw.text((indent_x, y_pos), tline, fill=(0, 100, 200), font=font_trans)
                y_pos += line_spacing

    return img


def _draw_cover_title(frame_rgb, title, max_w_ratio=0.88, base_font_size=80):
    """Draw a centered, word-wrapped title over the frame.
    Each line gets a solid black rounded-rectangle background pill.
    White text with thin black outline on top, text is centered within the pill."""
    img = Image.fromarray(frame_rgb.copy())
    fw, fh = img.size
    max_w = int(fw * max_w_ratio)

    # Find the largest font size where every wrapped line fits within max_w
    for fsize in range(base_font_size, 28, -4):
        font  = _font_for_lang('en', fsize)
        _tmp_draw = ImageDraw.Draw(img)
        lines = _wrap_text_to_lines(title, font, _tmp_draw, max_w, lang='en')
        if all(
            (_tmp_draw.textbbox((0, 0), ln, font=font)[2] -
             _tmp_draw.textbbox((0, 0), ln, font=font)[0]) <= max_w
            for ln in lines
        ):
            break

    # Measure each line's glyph bounding box
    tmp_draw = ImageDraw.Draw(img)
    bbs = [tmp_draw.textbbox((0, 0), ln, font=font) for ln in lines]
    lws = [b[2] - b[0] for b in bbs]   # glyph pixel width
    lhs = [b[3] - b[1] for b in bbs]   # glyph pixel height

    pad_x  = max(24, fsize // 2)        # horizontal padding inside pill
    pad_y  = max(16, fsize // 4)        # vertical padding inside pill
    gap    = max(10, fsize // 8)        # gap between pills
    radius = max(16, fsize // 3)        # corner radius

    # Total block height: each pill = glyph_h + 2*pad_y, separated by gap
    pill_hs = [lh + 2 * pad_y for lh in lhs]
    total_h = sum(pill_hs) + gap * max(0, len(lines) - 1)

    # Build RGBA overlay for all pill backgrounds
    overlay = Image.new('RGBA', (fw, fh), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)

    pill_y = (fh - total_h) // 2   # top of first pill
    pill_rects = []
    for i, line in enumerate(lines):
        # Pill rect — exactly wraps the glyph bbox + padding on all four sides
        pill_x1 = (fw - lws[i]) // 2 - pad_x
        pill_y1 = pill_y
        pill_x2 = (fw - lws[i]) // 2 + lws[i] + pad_x
        pill_y2 = pill_y + pill_hs[i]
        ov_draw.rounded_rectangle([pill_x1, pill_y1, pill_x2, pill_y2],
                                   radius=radius, fill=(0, 0, 0, 255))

        # Text draw origin: glyph top-left lands at (pill_x1+pad_x, pill_y1+pad_y)
        text_left = (fw - lws[i]) // 2
        text_top  = pill_y + pad_y
        draw_x = text_left - bbs[i][0]
        draw_y = text_top  - bbs[i][1]
        pill_rects.append((draw_x, draw_y, line))

        pill_y += pill_hs[i] + gap

    # Composite solid pill backgrounds onto frame
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(img)

    # Draw text: thin black outline then white fill
    outline = max(1, fsize // 30)
    for draw_x, draw_y, line in pill_rects:
        for dx in range(-outline, outline + 1):
            for dy in range(-outline, outline + 1):
                if dx != 0 or dy != 0:
                    draw.text((draw_x + dx, draw_y + dy), line, fill=(0, 0, 0), font=font)
        draw.text((draw_x, draw_y), line, fill=(255, 255, 255), font=font)

    return np.array(img)


def create_method4_video(input_vdo, output_path, src_srt_path, dst_srt_path, vocab_list,
                         audio_source=None, input_lang='en', output_lang='th',
                         video_size='tiktok', ep_label=None, cover_frame=None,
                         cover_title=None, log_func=None):
    """Method 4: video + bilingual subtitle bar (input lang top / output lang bottom) + vocab panel.
    video_size: 'tiktok' (1080×1920 portrait) or 'youtube' (1920×1080 landscape).
    Audio is muxed via ffmpeg stream copy — no re-encoding, 100% original quality."""
    import subprocess
    def log(m):
        if log_func: log_func(m)
        else: print(m)

    if video_size == 'youtube':
        # YouTube 16:9 landscape layout
        canvas_w, canvas_h = 1920, 1080
        LEFT_W    = 1100   # video + subtitle section (left)
        RIGHT_W   = canvas_w - LEFT_W  # 820, vocab section (right)
        sub_bar_h = 280    # subtitle bar height (4 slots × 70px)
        video_h   = canvas_h - sub_bar_h  # 800
        sub_y     = video_h
        vocab_x   = LEFT_W
        sub_w     = LEFT_W   # subtitle spans only the left section
        SLOT_H    = sub_bar_h // 4  # 70px
    else:
        # TikTok portrait layout (default)
        canvas_w, canvas_h = 1080, 1920
        LEFT_W    = canvas_w
        RIGHT_W   = 0
        sub_bar_h = 360    # 4 slots × 90px
        video_h   = 820
        vocab_h   = canvas_h - video_h - sub_bar_h  # 740
        sub_y     = video_h
        vocab_x   = 0
        sub_w     = canvas_w
        SLOT_H    = sub_bar_h // 4  # 90px

    # --- Use raw video for audio (anime video has no audio track) ---
    src_audio = audio_source if (audio_source and os.path.exists(audio_source)) else input_vdo
    log(f"🎬 [Method4] Original audio source: {os.path.basename(src_audio)}")
    render_duration = VideoFileClip(src_audio).duration  # duration only — audio muxed later via ffmpeg

    # --- Video (fills left/full section) ---
    log("🎬 [Method4] Loading content video...")
    content_raw = VideoFileClip(input_vdo).without_audio().set_duration(render_duration)
    if content_raw.w / content_raw.h < sub_w / video_h:
        content = content_raw.resize(width=sub_w)
    else:
        content = content_raw.resize(height=video_h)
    content = content.crop(x_center=content.w / 2, y_center=content.h / 2,
                            width=sub_w, height=video_h).set_duration(render_duration)

    # Override first frame with cover_frame + optional title overlay (shared across all EPs)
    if cover_frame is not None:
        from PIL import Image as _PILI
        _cf = np.array(_PILI.fromarray(cover_frame).resize((content.w, content.h)))
        if cover_title:
            _cf = _draw_cover_title(_cf, cover_title)
        _frame_dur = 1 / 30
        _cf_captured = _cf  # capture for closure
        content = content.fl(lambda gf, t, _f=_cf_captured: _f if t < _frame_dur else gf(t),
                              keep_duration=True)

    # --- Vocab panel (static) ---
    log("🎬 [Method4] Rendering vocab panel...")
    highlight_words = [e.get('word', '') for e in vocab_list]
    if video_size == 'youtube':
        vocab_img = _create_m4_vocab_image(vocab_list, RIGHT_W, canvas_h,
                                           src_lang=input_lang, dst_lang=output_lang,
                                           top_padding_lines=1)
        vocab_clip = (ImageClip(np.array(vocab_img))
                      .set_duration(render_duration).set_position((vocab_x, 0)))
    else:
        vocab_h = canvas_h - video_h - sub_bar_h
        vocab_img = _create_m4_vocab_image(vocab_list, canvas_w, vocab_h,
                                           src_lang=input_lang, dst_lang=output_lang)
        vocab_clip = (ImageClip(np.array(vocab_img))
                      .set_duration(render_duration).set_position((0, video_h + sub_bar_h)))

    # --- Load both SRTs and pair by index ---
    log(f"🎬 [Method4] Building subtitles: top={input_lang}, bottom={output_lang}...")
    src_subs = pysubs2.load(src_srt_path)
    dst_subs = pysubs2.load(dst_srt_path) if dst_srt_path and os.path.exists(dst_srt_path) else []

    # Temporary draw surface for pixel-accurate text measurement
    _tmp  = Image.new('RGB', (sub_w, sub_bar_h))
    _draw = ImageDraw.Draw(_tmp)
    max_w = sub_w - 60

    BASE_SRC = 52
    BASE_DST = 50
    MAX_LINES = 2

    sub_clips = []
    for i, ssub in enumerate(src_subs):
        src_text = ssub.text.strip().replace('\\N', ' ').replace('\n', ' ')
        dst_text = dst_subs[i].text.strip().replace('\\N', ' ').replace('\n', ' ') if i < len(dst_subs) else ''
        if not src_text:
            continue

        start_t = ssub.start / 1000.0
        end_t   = src_subs[i + 1].start / 1000.0 if i + 1 < len(src_subs) else ssub.end / 1000.0
        end_t   = min(end_t, render_duration)
        dur     = end_t - start_t
        if start_t >= render_duration or dur < 0.02:
            continue

        # Find the largest font size where both src and dst truly fit in MAX_LINES lines.
        # IMPORTANT: measure with max_lines=99 so the wrap count reflects the real line count,
        # not the capped result (which would always be ≤ 2 and never trigger a shrink).
        fs_src, fs_dst = BASE_SRC, BASE_DST
        src_lines, dst_lines = [], []
        for shrink in range(0, BASE_SRC - 18, 2):   # 52→20 in steps of 2
            fs_src = BASE_SRC - shrink
            fs_dst = BASE_DST - shrink
            f_src = _font_for_lang(input_lang,  fs_src)
            f_dst = _font_for_lang(output_lang, fs_dst)
            # Measure without cap to get true line count
            src_all = _wrap_for_lang(input_lang,  src_text, f_src, _draw, max_w, max_lines=99)
            dst_all = (_wrap_for_lang(output_lang, dst_text, f_dst, _draw, max_w, max_lines=99)
                       if dst_text else [])
            src_lines = src_all[:MAX_LINES]
            dst_lines = dst_all[:MAX_LINES]
            if len(src_all) <= MAX_LINES and len(dst_all) <= MAX_LINES:
                break   # text truly fits at this font size

        bar_np = np.array(_render_m4_subtitle_bar(src_lines, dst_lines, highlight_words,
                                                   sub_w, SLOT_H,
                                                   src_lang=input_lang, dst_lang=output_lang,
                                                   font_size_src=fs_src, font_size_dst=fs_dst))
        bar_h = bar_np.shape[0]
        y_pos = sub_y + (sub_bar_h - bar_h) // 2
        bg_clip  = (ColorClip(size=(sub_w, bar_h), color=(45, 45, 45))
                    .set_duration(dur).set_start(start_t).set_position((0, y_pos)))
        bar_clip = (ImageClip(bar_np).set_duration(dur).set_start(start_t).set_position((0, y_pos)))
        sub_clips.extend([bg_clip, bar_clip])

    log(f"🎬 [Method4] {len(sub_clips) // 2} subtitle clips created")

    # --- Composite video WITHOUT audio (audio will be muxed by ffmpeg) ---
    log("🎬 [Method4] Compositing (video only, no audio)...")
    bg_black = ColorClip(size=(canvas_w, canvas_h), color=(0, 0, 0)).set_duration(render_duration)
    if video_size == 'youtube':
        # YouTube: cream background for right vocab panel
        bg_cream = (ColorClip(size=(RIGHT_W, canvas_h), color=(245, 243, 235))
                    .set_duration(render_duration).set_position((vocab_x, 0)))
        layers = [bg_black, bg_cream, content.set_position((0, 0)), vocab_clip] + sub_clips
    else:
        layers = [bg_black, content.set_position((0, 0)), vocab_clip] + sub_clips

    # --- EP label badge ---
    # YouTube: top-left of canvas; TikTok: bottom-right of the video section
    if ep_label:
        _badge_font = _font_for_lang('en', 48)
        _tmp_img = Image.new('RGB', (400, 100), (0, 0, 0))
        _tmp_drw = ImageDraw.Draw(_tmp_img)
        _bb = _tmp_drw.textbbox((0, 0), ep_label, font=_badge_font)
        _pad_x, _pad_y = 20, 12
        _tw = _bb[2] - _bb[0]
        _th = _bb[3] - _bb[1]
        _bw = _tw + _pad_x * 2
        _bh = _th + _pad_y * 2
        _badge_img = Image.new('RGB', (_bw, _bh), (210, 30, 30))
        _badge_drw = ImageDraw.Draw(_badge_img)
        _badge_drw.text((_pad_x - _bb[0], _pad_y - _bb[1]), ep_label,
                        fill=(255, 255, 255), font=_badge_font)
        if video_size == 'tiktok':
            # Bottom-left of the video section (above the subtitle bar)
            _bx = 14
            _by = video_h - _bh - 14
        else:
            # Top-left of canvas for YouTube
            _bx, _by = 14, 14
        badge_clip = (ImageClip(np.array(_badge_img))
                      .set_duration(render_duration).set_position((_bx, _by)))
        layers.append(badge_clip)

    final = CompositeVideoClip(layers, size=(canvas_w, canvas_h))
    # No .set_audio() — audio preserved via ffmpeg stream copy below

    temp_noaudio = output_path.replace('.mp4', '_noaudio.mp4')
    log(f"🎬 [Method4] Rendering video track: {os.path.basename(temp_noaudio)}...")
    final.write_videofile(temp_noaudio, fps=30, codec="libx264", audio=False, threads=4, logger=None)

    # --- Mux original audio via ffmpeg stream copy (100% original, no re-encoding) ---
    log(f"🎬 [Method4] Muxing original audio (ffmpeg stream copy)...")
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_noaudio,
        "-i", src_audio,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio mux failed:\n{result.stderr[-500:]}")
    os.remove(temp_noaudio)
    log("✅ Method4 composition complete.")


# --- 4. MAIN RUNNER ---

async def run_pipeline(video_url, playlist_name, use_voxcpm_tts=True, use_anime=True, video_method=1,
                       input_lang='en', output_lang='th', video_size='tiktok',
                       max_length=0, progress_callback=None):
    def log(msg):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    try:
        global WS_DIR, OUTPUT_DIR, RAW_VIDEO, ANIME_VIDEO, THAI_SUB, THAI_VOICE, DATA_JSON, ENG_NAME, USE_VOXCPM_TTS, USE_ANIME
        
        USE_VOXCPM_TTS = use_voxcpm_tts
        USE_ANIME = use_anime
        OUTPUT_DIR = f"outputs/{playlist_name}/{video_size}"

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
        if os.path.exists(job_ws_dir):
            shutil.rmtree(job_ws_dir)
            log(f"🧹 Cleared workspace: {job_ws_dir}")
        os.makedirs(job_ws_dir)

        raw_video_path = os.path.join(job_ws_dir, "input_video.mp4")

        # Step 1: Download
        if is_valid_video(raw_video_path):
            log("⏭️ Step 1: Video already downloaded & valid, skipping...")
        else:
            log("🚀 Step 1: Downloading source video...")
            r = await asyncio.to_thread(download_video_to_path, video_url, raw_video_path)
            if r['status'] == 'error':
                raise Exception(f"Download failed: {r['message']}")

        # Step 1b: Split at sentence boundaries when max_length is set
        import subprocess as _sp
        episodes = []  # list of (ep_num, total_eps, video_path, start_sec, end_sec)
        full_src_srt = os.path.join(job_ws_dir, "full_src.srt")
        full_dst_srt = os.path.join(job_ws_dir, "full_dst.srt")

        if max_length and max_length > 0:
            actual_dur = _ffprobe_duration(raw_video_path)
            if actual_dur > max_length:
                # Transcribe the full video once to locate sentence boundaries
                if not (is_valid_srt(full_src_srt) and is_valid_srt(full_dst_srt)):
                    log("✂️ Step 1b: Transcribing full video to find sentence-boundary split points...")
                    await asyncio.to_thread(generate_thai_sub_to_path, raw_video_path, full_dst_srt,
                                            job_ws_dir, log_func=log, eng_srt_path=full_src_srt,
                                            input_lang=input_lang, output_lang=output_lang)
                else:
                    log("⏭️ Step 1b: Full-video SRT already exists, reusing for split points...")

                segments = _find_sentence_split_points(full_src_srt, max_length, actual_dur)
                total_eps = len(segments)
                log(f"✂️ Step 1b: {total_eps} episodes at sentence boundaries (max {max_length}s each)")

                for _i, (_s, _e) in enumerate(segments):
                    _ep_path = os.path.join(job_ws_dir, f"input_ep{_i+1}.mp4")
                    if not os.path.exists(_ep_path):
                        _sp.run(["ffmpeg", "-y", "-i", raw_video_path,
                                 "-ss", str(_s), "-t", str(_e - _s),
                                 "-c", "copy", _ep_path], capture_output=True)
                    log(f"  ✂️ EP{_i+1}: {_s:.1f}s – {_e:.1f}s ({_e-_s:.1f}s)")
                    episodes.append((_i + 1, total_eps, _ep_path, _s, _e))
            else:
                log(f"⏭️ Step 1b: Video {actual_dur:.1f}s ≤ {max_length}s — no split")
                episodes = [(None, None, raw_video_path, None, None)]
        else:
            episodes = [(None, None, raw_video_path, None, None)]

        method_suffix = f"_m{video_method}"
        if use_voxcpm_tts:
            method_suffix += "_voxcpm"

        # Sequential job number (persisted per output folder)
        job_num    = _next_job_number(OUTPUT_DIR)
        job_prefix = f"{job_num:03d}"
        log(f"📂 Job number: {job_prefix}")

        # Extract shared cover frame from EP1 raw video (first non-dark frame)
        cover_frame = None
        if video_method == 4 and len(episodes) > 1:
            ep1_video = episodes[0][2]
            log("🖼️ Extracting cover frame from EP1...")
            cover_frame = await asyncio.to_thread(_find_first_bright_frame, ep1_video)
            if cover_frame is not None:
                log("✅ Cover frame extracted — will be used as first frame for all EPs")

        # Generate title & hashtags ONCE for all EPs (from full SRT when available)
        job_title    = None
        job_hashtags = []
        if video_method == 4:
            _title_srt = full_src_srt if os.path.exists(full_src_srt) else None
            if _title_srt:
                log("🤖 Generating shared title & hashtags from full transcript...")
                _meta = await asyncio.to_thread(generate_title_and_hashtags, _title_srt,
                                                input_lang=input_lang, output_lang=output_lang,
                                                log_func=log)
                job_title    = _meta.get('title')
                job_hashtags = _meta.get('hashtags', [])
            # else: will be generated inside the loop from EP1's SRT and reused

        last_vdo_path = None
        last_json_path = None

        for ep_num, total_eps, ep_raw_video, ep_start, ep_end in episodes:
            ep_tag    = f"_ep{ep_num}of{total_eps}" if ep_num else ""
            ep_label  = f"EP.{ep_num}/{total_eps}" if ep_num else None
            ep_ws     = os.path.join(job_ws_dir, f"ep{ep_num}") if ep_num else job_ws_dir
            os.makedirs(ep_ws, exist_ok=True)

            if ep_num:
                log(f"\n{'='*50}\n🎬 Processing {ep_label} ({ep_num}/{total_eps})\n{'='*50}")

            # Episode-specific intermediate paths
            ep_thai_sub    = os.path.join(ep_ws, "thai_sub.srt")
            ep_synced_sub  = os.path.join(ep_ws, "thai_sub_synced.srt")
            ep_eng_sub     = os.path.join(ep_ws, "eng_sub.srt")
            ep_thai_voice  = os.path.join(ep_ws, "thai_dub.mp3")
            ep_data_json   = os.path.join(ep_ws, "data.json")
            ep_vocab_json  = os.path.join(ep_ws, "vocab_m4.json")
            ep_anime_video = os.path.join(ep_ws, "input_video_anime.mp4")
            ep_video_name  = os.path.splitext(os.path.basename(ep_raw_video))[0]
            ep_bg_audio    = os.path.join(ep_ws, f"separated/htdemucs/{ep_video_name}/no_vocals.wav")
            ep_timing_json = ep_thai_voice.replace(".mp3", "_timing.json")

            # Step 2: Use pre-sliced SRTs from full-video transcription, or run Whisper per-episode
            if ep_start is not None and is_valid_srt(full_src_srt) and is_valid_srt(full_dst_srt):
                if is_valid_srt(ep_eng_sub) and is_valid_srt(ep_thai_sub):
                    log("⏭️ Step 2: Episode SRTs already extracted, skipping...")
                else:
                    log(f"⏭️ Step 2: Slicing SRT {ep_start:.1f}s–{ep_end:.1f}s from full transcription...")
                    full_src_subs = pysubs2.load(full_src_srt)
                    full_dst_subs = pysubs2.load(full_dst_srt)
                    _extract_srt_segment(full_src_subs, ep_start, ep_end, ep_eng_sub)
                    _extract_srt_segment(full_dst_subs, ep_start, ep_end, ep_thai_sub)
            else:
                ep_srt_missing_for_m4 = (video_method == 4 and not is_valid_srt(ep_eng_sub))
                if is_valid_srt(ep_thai_sub) and not ep_srt_missing_for_m4:
                    log("⏭️ Step 2: Subtitles already valid, skipping...")
                else:
                    log(f"🚀 Step 2: Transcribing ({input_lang}) & Translating → ({output_lang})...")
                    await asyncio.to_thread(generate_thai_sub_to_path, ep_raw_video, ep_thai_sub, ep_ws,
                                            log_func=log, eng_srt_path=ep_eng_sub,
                                            input_lang=input_lang, output_lang=output_lang)

            # Step 3: Separate Background Audio (skipped for Method 4)
            if video_method == 4:
                log("⏭️ Step 3: Skipping background separation (Method 4 uses original audio)")
            elif is_valid_audio(ep_bg_audio):
                log("⏭️ Step 3: Background audio already valid, skipping...")
            else:
                log("🚀 Step 3: Separating Background Audio...")
                await asyncio.to_thread(separate_bg_audio_to_path, ep_raw_video, ep_ws)

            # Step 4: Generate Thai Dubbing (skipped for Method 4)
            if video_method == 4:
                log("⏭️ Step 4: Skipping Thai dubbing (Method 4 uses original audio)")
            elif is_valid_audio(ep_thai_voice):
                log("⏭️ Step 4: Thai dubbing already valid, skipping...")
                if not is_valid_srt(ep_synced_sub) and os.path.exists(ep_timing_json):
                    with open(ep_timing_json, "r", encoding="utf-8") as f:
                        timing_data = json.load(f)
                    _write_synced_srt(timing_data, ep_synced_sub)
                    log("🔄 Step 4b: Rebuilt synced SRT from timing cache.")
            else:
                log("🚀 Step 4: Generating Thai Dubbing...")
                await make_final_audio_to_path(ep_thai_sub, ep_thai_voice, ep_ws, use_voxcpm_tts,
                                               log_func=log, synced_srt_path=ep_synced_sub)

            # Step 5: Anime (optional)
            if use_anime:
                if is_valid_video(ep_anime_video):
                    log("⏭️ Step 5: Anime video already valid, skipping...")
                else:
                    log("🎨 Step 5: Applying Anime Style Conversion...")
                    await asyncio.to_thread(run_anime_conversion_to_path, ep_raw_video, ep_anime_video, log_func=log)
                final_input_vdo = ep_anime_video
                suffix = "_anime"
            else:
                final_input_vdo = ep_raw_video
                suffix = ""

            # Step 6b: Extract vocabulary for Method 4 (per-episode, based on episode SRT)
            vocab_list = []
            if video_method == 4:
                if os.path.exists(ep_vocab_json):
                    log("⏭️ Step 6b: Vocab already extracted, skipping...")
                    with open(ep_vocab_json, 'r', encoding='utf-8') as f:
                        vocab_list = json.load(f)
                else:
                    log("🤖 Step 6b: Extracting vocabulary for M4 (episode-specific)...")
                    ep_dur_sec = (ep_end - ep_start) if (ep_end is not None and ep_start is not None) else _ffprobe_duration(ep_raw_video)
                    vocab_list = await asyncio.to_thread(extract_vocab_with_ollama, ep_eng_sub,
                                                         input_lang=input_lang, output_lang=output_lang,
                                                         clip_duration_sec=ep_dur_sec, log_func=log)
                    with open(ep_vocab_json, 'w', encoding='utf-8') as f:
                        json.dump(vocab_list, f, ensure_ascii=False, indent=2)

            # Step 6c: Title & hashtags — shared across all EPs (generate once from EP1 if needed)
            ep_title    = job_title
            ep_hashtags = job_hashtags
            if video_method == 4 and job_title is None:
                ep_meta_json = os.path.join(job_ws_dir, "meta.json")  # shared file, not per-ep
                if os.path.exists(ep_meta_json):
                    log("⏭️ Step 6c: Shared title/hashtags already cached, skipping...")
                    with open(ep_meta_json, 'r', encoding='utf-8') as f:
                        _meta = json.load(f)
                else:
                    log("🤖 Step 6c: Generating shared title & hashtags from EP1 SRT...")
                    _meta = await asyncio.to_thread(generate_title_and_hashtags, ep_eng_sub,
                                                    input_lang=input_lang, output_lang=output_lang,
                                                    log_func=log)
                    with open(ep_meta_json, 'w', encoding='utf-8') as f:
                        json.dump(_meta, f, ensure_ascii=False, indent=2)
                job_title    = _meta.get('title')
                job_hashtags = _meta.get('hashtags', [])
                ep_title     = job_title
                ep_hashtags  = job_hashtags

            # Step 6: AI Suggestions (Method 1 only)
            title_hook = 'สรุปคลิปที่น่าสนใจ'
            if video_method == 1:
                if is_valid_json(ep_data_json):
                    log("⏭️ Step 6: AI suggestions already valid, skipping...")
                    with open(ep_data_json, 'r', encoding='utf-8') as f:
                        ai_data = json.load(f)
                else:
                    log("🤖 Step 6: Getting AI Content Suggestions (Ollama/Llama3)...")
                    ai_data = {}
                    for _ in range(5):
                        ai_data = await asyncio.to_thread(get_ollama_suggestions_to_path, ep_thai_sub, ep_data_json, log_func=log)
                        is_valid_ai = all(v is not None for v in ai_data.values())
                        if is_valid_ai: break
                title_hook = ai_data.get('title_hook', 'สรุปคลิปที่น่าสนใจ')
            else:
                log(f"⏭️ Step 6: Skipping AI suggestions (Method {video_method} has no title)")

            # Filename: {prefix}_{title_slug}{_epXofY}  (job_prefix shared across all EPs)
            if video_method == 4 and ep_title:
                name_slug = _safe_filename(ep_title)
            else:
                name_slug = f"tk_{ENG_NAME}{suffix}{method_suffix}"
            base_name       = f"{job_prefix}_{name_slug}{ep_tag}"
            final_vdo_path  = unique_path(os.path.join(OUTPUT_DIR, f"{base_name}.mp4"))
            final_json_path = unique_path(os.path.join(OUTPUT_DIR, f"{base_name}.json"))
            final_base = final_vdo_path[:-4]  # strip ".mp4"

            sub_for_video = ep_synced_sub if os.path.exists(ep_synced_sub) else ep_thai_sub

            # Step 7: Final Composition
            if video_method == 2:
                log("🎬 Step 7: Compositing Method 2...")
                await asyncio.to_thread(create_method2_video, final_input_vdo, ep_thai_voice, final_vdo_path, sub_for_video, bg_audio_path=ep_bg_audio, log_func=log)
            elif video_method == 3:
                log("🎬 Step 7: Compositing Method 3...")
                await asyncio.to_thread(create_method3_video, final_input_vdo, ep_thai_voice, final_vdo_path, sub_for_video, bg_audio_path=ep_bg_audio, log_func=log)
            elif video_method == 4:
                log(f"🎬 Step 7: Compositing Method 4 ({video_size}){' — ' + ep_label if ep_label else ''}...")
                await asyncio.to_thread(create_method4_video, final_input_vdo, final_vdo_path,
                                        ep_eng_sub, ep_thai_sub, vocab_list,
                                        audio_source=ep_raw_video,
                                        input_lang=input_lang, output_lang=output_lang,
                                        video_size=video_size, ep_label=ep_label,
                                        cover_frame=cover_frame, cover_title=ep_title,
                                        log_func=log)
            else:
                log("🎬 Step 7: Compositing Method 1...")
                await asyncio.to_thread(create_final_tiktok_video, final_input_vdo, ep_thai_voice, final_vdo_path, title_hook, ep_thai_sub, bg_audio_path=ep_bg_audio, log_func=log)

            if os.path.exists(ep_data_json):
                shutil.copy(ep_data_json, final_json_path)

            # M4: save original clip, SRTs, vocab JSON with source_url
            if video_method == 4:
                vocab_with_meta = {
                    "source_url": video_url,
                    "title": ep_title,
                    "hashtags": ep_hashtags,
                    "vocab": vocab_list,
                }
                vocab_out_path = unique_path(final_base + "_vocab.json")
                with open(vocab_out_path, 'w', encoding='utf-8') as f:
                    json.dump(vocab_with_meta, f, ensure_ascii=False, indent=2)
                log(f"📄 Saved: {os.path.basename(vocab_out_path)}")
                for src, dst in [
                    (ep_raw_video, unique_path(final_base + "_original.mp4")),
                    (ep_eng_sub,   unique_path(final_base + "_eng.srt")),
                    (ep_thai_sub,  unique_path(final_base + "_thai.srt")),
                ]:
                    if os.path.exists(src):
                        shutil.copy(src, dst)
                        log(f"📄 Saved: {os.path.basename(dst)}")

            log(f"✅ {'[' + ep_label + '] ' if ep_label else ''}Output: {final_vdo_path}")
            last_vdo_path  = final_vdo_path
            last_json_path = final_json_path

        return {"status": "success", "video": os.path.abspath(last_vdo_path), "json": os.path.abspath(last_json_path)}

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
    parser.add_argument("--method", type=int, default=1)
    args = parser.parse_args()

    asyncio.run(run_pipeline(args.url, args.playlist, args.voxcpm, args.anime, args.method))