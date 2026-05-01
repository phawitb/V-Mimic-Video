from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import uuid
import json
import os
import asyncio
from datetime import datetime, time, timedelta
from run_all import run_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure outputs directory exists and serve it
if not os.path.exists("outputs"):
    os.makedirs("outputs")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# --- Persistence ---
STATE_FILE = "state.json"

class AppState:
    def __init__(self):
        self.queue = []      # List of job IDs in order
        self.jobs = {}       # Full job details by ID
        self.settings = {
            "start_time": "09:00",
            "end_time": "23:59",
            "is_enabled": True
        }
        self.active_job_id = None
        self.active_task = None  # To store the current asyncio task
        self.last_retry_date = None # Track the last date we did the auto-retry
        self.load_state()

    def save_state(self):
        # Run in thread to not block event loop
        import threading
        def _save():
            data = {
                "queue": self.queue,
                "jobs": self.jobs,
                "settings": self.settings,
                "last_retry_date": str(self.last_retry_date) if self.last_retry_date else None
            }
            try:
                with open(STATE_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"👷 Error saving state: {e}")
        threading.Thread(target=_save, daemon=True).start()

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.queue = data.get("queue", [])
                    self.jobs = data.get("jobs", {})
                    self.settings = data.get("settings", self.settings)
                    lrd = data.get("last_retry_date")
                    if lrd and lrd != "None":
                        self.last_retry_date = datetime.strptime(lrd, "%Y-%m-%d").date()
                    
                    # Reset any 'processing' jobs to 'queued' on startup
                    for job_id, job in self.jobs.items():
                        if job["status"] == "processing":
                            job.pop("error", None) # Clear old error
                            job["status"] = "queued"
                            if job_id not in self.queue:
                                self.queue.insert(0, job_id)
            except Exception as e:
                print(f"👷 Error loading state: {e}")

state = AppState()

class AddLinksRequest(BaseModel):
    video_urls: List[str]
    playlist_name: str
    use_voxcpm_tts: bool = True
    use_anime: bool = True
    video_method: int = 1
    input_lang: str = 'en'
    output_lang: str = 'th'
    video_size: str = 'tiktok'
    max_length: int = 0

class UpdateJobRequest(BaseModel):
    playlist_name: Optional[str] = None
    use_voxcpm_tts: Optional[bool] = None
    use_anime: Optional[bool] = None
    video_method: Optional[int] = None
    input_lang: Optional[str] = None
    output_lang: Optional[str] = None
    video_size: Optional[str] = None
    max_length: Optional[int] = None

class SettingsRequest(BaseModel):
    start_time: str
    end_time: str
    is_enabled: bool

@app.get("/")
async def read_index():
    return FileResponse("index.html")

@app.get("/status")
async def get_status():
    return {
        "settings": state.settings,
        "queue_count": len(state.queue),
        "active_job_id": state.active_job_id,
        "is_working_now": is_within_working_period()
    }

@app.get("/jobs")
async def get_jobs():
    # Return jobs WITHOUT logs to keep payload small
    summary = []
    for jid, job in state.jobs.items():
        job_copy = job.copy()
        job_copy.pop("logs", None)
        summary.append(job_copy)
    summary.sort(key=lambda x: x["created_at"], reverse=True)
    return summary

@app.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return state.jobs[job_id].get("logs", [])

@app.post("/links")
async def add_links(req: AddLinksRequest):
    new_job_ids = []
    for url in req.video_urls:
        if not url.strip(): continue
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "video_url": url.strip(),
            "playlist_name": req.playlist_name,
            "use_voxcpm_tts": req.use_voxcpm_tts,
            "use_anime": req.use_anime,
            "video_method": req.video_method,
            "input_lang": req.input_lang,
            "output_lang": req.output_lang,
            "video_size": req.video_size,
            "max_length": req.max_length,
            "status": "queued",
            "logs": [],
            "created_at": datetime.now().isoformat()
        }
        state.jobs[job_id] = job
        state.queue.append(job_id)
        new_job_ids.append(job_id)
    state.save_state()
    return {"added": len(new_job_ids)}

@app.post("/jobs/{job_id}/run")
async def manual_run(job_id: str):
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if state.active_job_id == job_id:
        raise HTTPException(status_code=400, detail="Job is already running")
    
    if job_id in state.queue:
        state.queue.remove(job_id)
    
    job = state.jobs[job_id]
    job["status"] = "queued"
    job["logs"] = [] # Clear logs on re-run
    job.pop("error", None) # Clear old error
    
    state.queue.insert(0, job_id)
    state.save_state()
    return {"status": "moved_to_front"}

@app.post("/jobs/{job_id}/interrupt")
async def interrupt_job(job_id: str):
    if state.active_job_id == job_id and state.active_task:
        state.active_task.cancel()
        state.jobs[job_id]["status"] = "failed"
        state.jobs[job_id]["error"] = "Interrupted by user"
        state.jobs[job_id]["logs"].append({
            "timestamp": datetime.now().isoformat(),
            "message": "🛑 Job interrupted by user"
        })
        state.save_state()
        return {"status": "interrupted"}
    raise HTTPException(status_code=400, detail="Job is not active")

@app.get("/open-file")
async def open_file(path: str):
    import subprocess
    import platform
    abs_path = os.path.abspath(path)
    
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        if platform.system() == "Darwin":
            subprocess.run(["open", abs_path])
        elif platform.system() == "Windows":
            os.startfile(abs_path)
        else:
            subprocess.run(["xdg-open", abs_path])
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/open-folder")
async def open_folder(path: str):
    import subprocess
    import platform
    # ... (rest of function unchanged)
    abs_path = os.path.abspath(path)
    if os.path.isfile(abs_path):
        abs_path = os.path.dirname(abs_path)
    
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail="Folder not found")
        
    try:
        if platform.system() == "Darwin":
            subprocess.run(["open", abs_path])
        elif platform.system() == "Windows":
            os.startfile(abs_path)
        else:
            subprocess.run(["xdg-open", abs_path])
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/jobs/{job_id}")
async def update_job(job_id: str, req: UpdateJobRequest):
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = state.jobs[job_id]
    if job["status"] == "processing":
        raise HTTPException(status_code=400, detail="Cannot edit processing job")
    
    if req.playlist_name is not None: job["playlist_name"] = req.playlist_name
    if req.use_voxcpm_tts is not None: job["use_voxcpm_tts"] = req.use_voxcpm_tts
    if req.use_anime is not None: job["use_anime"] = req.use_anime
    if req.video_method is not None: job["video_method"] = req.video_method
    if req.input_lang is not None: job["input_lang"] = req.input_lang
    if req.output_lang is not None: job["output_lang"] = req.output_lang
    if req.video_size is not None: job["video_size"] = req.video_size
    if req.max_length is not None: job["max_length"] = req.max_length
    state.save_state()
    return job

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if job_id == state.active_job_id:
        raise HTTPException(status_code=400, detail="Cannot delete active job")
    
    if job_id in state.queue:
        state.queue.remove(job_id)
    del state.jobs[job_id]
    state.save_state()
    return {"status": "deleted"}

@app.post("/settings")
async def update_settings(req: SettingsRequest):
    state.settings.update(req.dict())
    state.save_state()
    return state.settings

def is_within_working_period():
    if not state.settings["is_enabled"]: return False
    now = datetime.now().time()
    try:
        start = datetime.strptime(state.settings["start_time"], "%H:%M").time()
        end = datetime.strptime(state.settings["end_time"], "%H:%M").time()
        if start <= end: return start <= now <= end
        return now >= start or now <= end
    except: return False

async def process_next_job():
    if state.active_job_id:
        return

    # Daily retry logic
    current_date = datetime.now().date()
    if is_within_working_period():
        if state.last_retry_date != current_date:
            # First time in working period for this date
            failed_ids = [jid for jid, job in state.jobs.items() if job["status"] == "failed"]
            if failed_ids:
                for jid in failed_ids:
                    if jid not in state.queue:
                        state.jobs[jid]["status"] = "queued"
                        state.jobs[jid]["logs"].append({
                            "timestamp": datetime.now().isoformat(),
                            "message": "♻️ New day retry: moving to end of queue"
                        })
                        state.queue.append(jid)
            state.last_retry_date = current_date
            state.save_state()

    if not state.queue:
        if is_within_working_period():
            # If queue is empty but we are in working period, look for failed jobs to retry
            failed_ids = [jid for jid, job in state.jobs.items() if job.get("status") == "failed"]
            if failed_ids:
                for jid in failed_ids:
                    state.jobs[jid]["status"] = "queued"
                    state.jobs[jid]["logs"].append({
                        "timestamp": datetime.now().isoformat(),
                        "message": "🔄 Auto-retrying failed job during working hours"
                    })
                    state.queue.append(jid)
                state.save_state()
            else: return
        else: return

    job_id = state.queue.pop(0)
    state.active_job_id = job_id
    state.active_task = asyncio.current_task() # Track current task for interruption
    job = state.jobs[job_id]
    job["status"] = "processing"
    state.save_state()
    
    # Clear very long logs
    if len(job["logs"]) > 100:
        job["logs"] = [{"timestamp": datetime.now().isoformat(), "message": "--- Resuming Job (Log Cleared) ---"}]

    def progress_callback(msg):
        # We wrap this to avoid list modification issues if needed
        job["logs"].append({"timestamp": datetime.now().isoformat(), "message": msg})
    
    try:
        # We run the pipeline. Note: if run_pipeline blocks, the loop freezes.
        # We'll rely on its internal awaits or wrap it if needed.
        result = await run_pipeline(
            job["video_url"],
            job["playlist_name"],
            job["use_voxcpm_tts"],
            job["use_anime"],
            job.get("video_method", 1),
            input_lang=job.get("input_lang", "en"),
            output_lang=job.get("output_lang", "th"),
            video_size=job.get("video_size", "tiktok"),
            max_length=job.get("max_length", 0),
            progress_callback=progress_callback
        )
        
        if result["status"] == "success":
            job["status"] = "completed"
            job["video_full_path"] = result["video"] # Store absolute path
            # Extract relative path for web serving
            # absolute path: /Users/.../outputs/playlist/file.mp4
            # we want: /outputs/playlist/file.mp4
            abs_path = result["video"]
            rel_path = abs_path.split("/outputs/")[-1]
            job["video_url_path"] = f"/outputs/{rel_path}"
        elif result["status"] == "error":
            job["status"] = "failed"
            job["error"] = result.get("message", "Unknown error")
    except asyncio.CancelledError:
        job["status"] = "failed"
        job["error"] = "Interrupted by user"
        # Re-raise to let worker_loop know it was cancelled if needed, 
        # but here we want to just stop this job and move to next.
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
    finally:
        state.active_job_id = None
        state.active_task = None
        state.save_state()

async def worker_loop():
    print("👷 Worker loop started")
    while True:
        try:
            await process_next_job()
        except Exception as e:
            print(f"👷 Worker error: {e}")
        await asyncio.sleep(5)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(worker_loop())

if __name__ == "__main__":
    import uvicorn
    # Using 1 worker to ensure the shared 'state' works as expected
    uvicorn.run(app, host="0.0.0.0", port=8123, workers=1)
