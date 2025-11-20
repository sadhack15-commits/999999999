# ==========================
# Discord bot + Ollama (OPTIMIZED FOR 512MB RAM)
# Model: qwen2.5:0.5b (Tiny model for low RAM)
# ==========================
import os, sys, subprocess, time, asyncio, json, re, traceback, zipfile, io, base64, socket
from pathlib import Path
from threading import Thread

# ---- Quick setup (pip + ollama) ----
def pip_install(pkgs):
    print("[pip] Installing:", " ".join(pkgs))
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", *pkgs], check=True)

# GIẢM DEPENDENCIES - chỉ cài những thứ cần thiết
pip_install([
    "discord.py>=2.4.0",
    "aiohttp>=3.9.5",
    "requests>=2.31.0",
    "nest_asyncio>=1.6.0",
    "flask>=3.0.0"
    # BỎ: scikit-learn (100MB), tensorflow (nặng)
])

import nest_asyncio
import requests
nest_asyncio.apply()

# ---- Helper: Tìm cổng khả dụng ----
def find_free_port(ports_to_try):
    for port in ports_to_try:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', port))
                print(f"✓ Found free port: {port}")
                return port
        except OSError as e:
            print(f"✗ Port {port} unavailable: {e}")
            continue
    raise RuntimeError(f"Cannot find any free port from: {ports_to_try}")

def find_free_ollama_port():
    return find_free_port([11434, 11475])

# ---- Tìm cổng ngay từ đầu ----
OLLAMA_PORT = find_free_ollama_port()

preferred_port = os.environ.get("PORT")
if preferred_port:
    try:
        WEB_PORT = find_free_port([int(preferred_port)])
    except:
        WEB_PORT = find_free_port([8080, 80, 9000])
else:
    WEB_PORT = find_free_port([8080, 80, 9000])

print(f"=== PORT CONFIGURATION ===")
print(f"Web Server Port: {WEB_PORT}")
print(f"Ollama Port: {OLLAMA_PORT}")
print(f"==========================\n")

# ---- Cài Ollama (NO SUDO - user mode) ----
print("[ollama] Installing in user mode...")

OLLAMA_DIR = Path.home() / ".ollama"
OLLAMA_BIN = OLLAMA_DIR / "bin"
OLLAMA_EXEC = OLLAMA_BIN / "ollama"

def install_ollama_user():
    if OLLAMA_EXEC.exists():
        print(f"✓ Ollama đã có tại: {OLLAMA_EXEC}")
        return
    
    OLLAMA_BIN.mkdir(parents=True, exist_ok=True)
    
    import platform
    import tarfile
    
    system = platform.system().lower()
    arch = platform.machine().lower()
    
    if "x86_64" in arch or "amd64" in arch:
        arch = "amd64"
    elif "aarch64" in arch or "arm64" in arch:
        arch = "arm64"
    else:
        raise RuntimeError(f"Unsupported architecture: {arch}")
    
    versions_to_try = ["v0.6.5", "v0.5.14", "v0.5.10"]
    
    last_error = None
    for idx, version in enumerate(versions_to_try, 1):
        try:
            filename = f"ollama-{system}-{arch}.tgz"
            url = f"https://github.com/ollama/ollama/releases/download/{version}/{filename}"
            
            print(f"\n[{idx}/{len(versions_to_try)}] Trying {version}: {url}")
            
            response = requests.get(
                url,
                stream=True,
                timeout=300,
                allow_redirects=True,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            
            if response.status_code != 200:
                print(f"  ✗ HTTP {response.status_code}")
                continue
            
            total_size = int(response.headers.get('content-length', 0))
            if total_size == 0:
                print(f"  ✗ Empty or unknown size")
                continue
            
            print(f"  ✓ Found! Downloading {total_size/1048576:.1f}MB...")
            
            tgz_path = OLLAMA_BIN / filename
            downloaded = 0
            last_print = 0
            
            with open(tgz_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if downloaded - last_print >= 5*1024*1024:
                            pct = (downloaded / total_size) * 100
                            print(f"    {pct:.0f}% ({downloaded/1048576:.0f}MB/{total_size/1048576:.0f}MB)")
                            last_print = downloaded
            
            print(f"  ✓ Downloaded: {tgz_path.stat().st_size/1048576:.1f}MB")
            print(f"  Extracting...")
            
            try:
                with tarfile.open(tgz_path, 'r:gz') as tar:
                    members = tar.getmembers()
                    ollama_member = None
                    
                    for member in members:
                        if member.name.endswith('/ollama') or member.name == 'ollama' or member.name.endswith('/bin/ollama'):
                            ollama_member = member
                            break
                    
                    if not ollama_member:
                        print(f"  ✗ No ollama binary found in tarball")
                        tgz_path.unlink()
                        continue
                    
                    tar.extract(ollama_member, path=OLLAMA_BIN)
                    extracted_path = OLLAMA_BIN / ollama_member.name
                    
                    if extracted_path != OLLAMA_EXEC:
                        extracted_path.rename(OLLAMA_EXEC)
                    
                    OLLAMA_EXEC.chmod(0o755)
                
                tgz_path.unlink()
                
                file_size = OLLAMA_EXEC.stat().st_size
                if file_size < 10*1024*1024:
                    print(f"  ✗ Binary too small ({file_size/1048576:.1f}MB)")
                    OLLAMA_EXEC.unlink()
                    continue
                
                print(f"✓ Ollama {version} installed: {OLLAMA_EXEC} ({file_size/1048576:.1f}MB)")
                return
                
            except tarfile.TarError as e:
                print(f"  ✗ Extract error: {e}")
                last_error = str(e)
                if tgz_path.exists():
                    tgz_path.unlink()
                if OLLAMA_EXEC.exists():
                    OLLAMA_EXEC.unlink()
                continue
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            last_error = str(e)
    
    raise RuntimeError(f"Failed to install Ollama. Last error: {last_error}")

install_ollama_user()
os.environ["PATH"] = f"{OLLAMA_BIN}:{os.environ['PATH']}"

# ⚠️ CRITICAL: Model nhỏ cho 512MB RAM
MODEL = "qwen2.5:0.5b"  # Model 500M parameters, chỉ ~350MB RAM
WARMUP_PROMPT = "hi"
BOT_NICKNAME = "Victory_vn_AI_✨"
REQ_TIMEOUT = 180  # Giảm timeout
DISCORD_MAX = 1900
HISTORY_TURNS = 2  # Giảm history để tiết kiệm RAM
MAX_ACTIVE_USERS = 1  # CHỈ 1 user đồng thời cho 512MB
BRAIN_MAX_CHARS = 0
RAG_TOPK = 1  # Giảm RAG
RAG_CHARS = 200
PREFERRED_PROFILE = "L3-ultra"  # Dùng profile nhỏ nhất

# ===== RAM MONITORING =====
RAM_LIMIT_MB = 512
RAM_WARNING_MB = 400  # Cảnh báo khi đạt 400MB (78%)
RAM_CRITICAL_MB = 480  # Nguy hiểm khi đạt 480MB (94%)
RAM_CHECK_INTERVAL = 30  # Kiểm tra mỗi 30s
RAM_HISTORY = []  # Lưu lịch sử RAM
MAX_RAM_HISTORY = 100

def get_memory_usage():
    """Lấy thông tin RAM usage (MB)"""
    try:
        # Đọc từ /proc/self/status (chính xác nhất cho process hiện tại)
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):  # Resident Set Size (RAM thực tế)
                    rss_kb = int(line.split()[1])
                    return rss_kb / 1024  # Convert KB -> MB
        
        # Fallback: dùng /proc/meminfo
        with open('/proc/meminfo', 'r') as f:
            meminfo = {}
            for line in f:
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = int(parts[1].strip().split()[0])
                    meminfo[key] = value / 1024  # KB -> MB
            
            used = meminfo.get('MemTotal', 0) - meminfo.get('MemAvailable', 0)
            return used
    except:
        # Fallback cuối: dùng psutil nếu có
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0

def check_ram_and_cleanup():
    """Kiểm tra RAM và cleanup nếu cần"""
    current_mb = get_memory_usage()
    timestamp = time.time()
    
    # Lưu lịch sử
    RAM_HISTORY.append({"ts": timestamp, "mb": current_mb})
    if len(RAM_HISTORY) > MAX_RAM_HISTORY:
        RAM_HISTORY.pop(0)
    
    # Cập nhật STATUS
    STATUS["ram_mb"] = current_mb
    STATUS["ram_pct"] = (current_mb / RAM_LIMIT_MB) * 100
    
    # Cảnh báo
    if current_mb >= RAM_CRITICAL_MB:
        print(f"🔴 CRITICAL RAM: {current_mb:.1f}MB / {RAM_LIMIT_MB}MB ({STATUS['ram_pct']:.1f}%)")
        # Emergency cleanup
        emergency_cleanup()
        return "critical"
    elif current_mb >= RAM_WARNING_MB:
        print(f"🟡 WARNING RAM: {current_mb:.1f}MB / {RAM_LIMIT_MB}MB ({STATUS['ram_pct']:.1f}%)")
        # Soft cleanup
        soft_cleanup()
        return "warning"
    else:
        return "ok"

def soft_cleanup():
    """Cleanup nhẹ: xóa history cũ, giới hạn KB"""
    try:
        # Xóa history channels cũ (giữ 5 mới nhất)
        if HIST_DIR.exists():
            files = sorted(HIST_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for f in files[5:]:
                f.unlink()
        
        # Giới hạn KB size
        if KB_FILE.exists():
            lines = KB_FILE.read_text().splitlines()
            if len(lines) > 50:  # Giữ 50 mục mới nhất
                KB_FILE.write_text("\n".join(lines[-50:]) + "\n")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print(f"✓ Soft cleanup completed")
    except Exception as e:
        print(f"✗ Soft cleanup failed: {e}")

def emergency_cleanup():
    """Cleanup khẩn cấp: xóa tất cả cache, restart Ollama"""
    try:
        print("🚨 EMERGENCY CLEANUP STARTED")
        
        # 1. Clear tất cả history
        if HIST_DIR.exists():
            for f in HIST_DIR.glob("*.json"):
                f.unlink()
        
        # 2. Clear KB
        if KB_FILE.exists():
            KB_FILE.write_text("")
        
        # 3. Force GC nhiều lần
        import gc
        for _ in range(3):
            gc.collect()
        
        # 4. Kill và restart Ollama (giải phóng model cache)
        try:
            subprocess.run(["pkill", "-9", "-f", "ollama serve"], check=False)
            time.sleep(2)
            
            ollama_env = os.environ.copy()
            ollama_env["OLLAMA_HOST"] = f"127.0.0.1:{OLLAMA_PORT}"
            ollama_env["OLLAMA_MODELS"] = str(DATA_DIR / "models")
            ollama_env["OLLAMA_KEEP_ALIVE"] = "5m"
            ollama_env["OLLAMA_NUM_PARALLEL"] = "1"
            ollama_env["OLLAMA_MAX_LOADED_MODELS"] = "1"
            ollama_env["OLLAMA_MAX_VRAM"] = "200m"  # Giảm xuống 200MB
            
            subprocess.Popen(
                [str(OLLAMA_EXEC), "serve"],
                stdout=open(OLLAMA_LOG, 'a'),
                stderr=subprocess.STDOUT,
                env=ollama_env
            )
            print("✓ Ollama restarted")
        except Exception as e:
            print(f"✗ Ollama restart failed: {e}")
        
        print("✓ Emergency cleanup completed")
    except Exception as e:
        print(f"✗ Emergency cleanup failed: {e}")

async def ram_monitor_loop():
    """Background task để monitor RAM liên tục"""
    await asyncio.sleep(10)  # Đợi bot khởi động
    
    while True:
        try:
            status = check_ram_and_cleanup()
            
            # Log mỗi 5 phút
            if int(time.time()) % 300 < RAM_CHECK_INTERVAL:
                current_mb = STATUS.get("ram_mb", 0)
                avg_mb = sum(h["mb"] for h in RAM_HISTORY[-10:]) / min(10, len(RAM_HISTORY)) if RAM_HISTORY else 0
                print(f"📊 RAM: Current={current_mb:.1f}MB, Avg(10)={avg_mb:.1f}MB, Status={status}")
            
        except Exception as e:
            print(f"RAM monitor error: {e}")
        
        await asyncio.sleep(RAM_CHECK_INTERVAL)

# ---- STATUS ----
STATUS = {
    "phase": "starting",
    "model": MODEL,
    "pull_pct": 0,
    "pull_layer": "",
    "pull_bytes": (0, 0),
    "last_err": "",
    "avg_sec": 0.0,
    "count": 0,
    "uptime": 0,
    "web_port": WEB_PORT,
    "ollama_port": OLLAMA_PORT,
    "ram_mb": 0,
    "ram_pct": 0,
    "ram_status": "ok"
}

def _hdr(title): print("\n" + "="*80 + f"\n[ {title} ]\n" + "="*80)

# ---- Data dirs ----
DATA_DIR = Path.home() / "bot_data"
BRAIN_DIR = DATA_DIR / "brain" / "users"
HIST_DIR = DATA_DIR / "history" / "channels"
CONF_FILE = DATA_DIR / "config.json"
KB_FILE = DATA_DIR / "kb.jsonl"
OLLAMA_LOG = DATA_DIR / "logs" / "ollama.log"

for p in [BRAIN_DIR, HIST_DIR, OLLAMA_LOG.parent]:
    p.mkdir(parents=True, exist_ok=True)

if not CONF_FILE.exists():
    CONF_FILE.write_text(json.dumps({"auto_channels": [], "auto_train": True}, ensure_ascii=False, indent=2))
if not KB_FILE.exists():
    KB_FILE.write_text("")

# ---- Kill tiến trình cũ ----
def run_cmd(cmd, desc=None, env=None, allow_fail=False):
    if desc: _hdr(desc)
    print(f"$ {' '.join(cmd)}")
    
    if env is None:
        env = os.environ.copy()
    if 'ollama' in ' '.join(cmd):
        env["OLLAMA_HOST"] = f"127.0.0.1:{OLLAMA_PORT}"
    
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env=env)
    for line in p.stdout:
        print(line, end="")
    ret = p.wait()
    if ret != 0 and not allow_fail:
        raise RuntimeError(f"Command failed (exit={ret}): {' '.join(cmd)}")
    return ret

run_cmd(["pkill", "-f", "ollama serve"], "Kill ollama cũ", allow_fail=True)

# ---- Start Ollama với cấu hình MINIMAL ----
_hdr("Start Ollama (MINIMAL CONFIG)")
OLLAMA_LOG.write_text("")
ollama_env = os.environ.copy()
ollama_env["OLLAMA_HOST"] = f"127.0.0.1:{OLLAMA_PORT}"
ollama_env["OLLAMA_MODELS"] = str(DATA_DIR / "models")
ollama_env["OLLAMA_KEEP_ALIVE"] = "10m"  # Giảm từ 24h
ollama_env["OLLAMA_NUM_PARALLEL"] = "1"  # CHỈ 1 request đồng thời
ollama_env["OLLAMA_MAX_LOADED_MODELS"] = "1"
# Giới hạn RAM cho Ollama
ollama_env["OLLAMA_MAX_VRAM"] = "256m"  # Chỉ dùng 256MB

subprocess.Popen(
    [str(OLLAMA_EXEC), "serve"],
    stdout=open(OLLAMA_LOG, 'w'),
    stderr=subprocess.STDOUT,
    env=ollama_env
)
print(f"Ollama đang khởi động trên cổng {OLLAMA_PORT}...")

import aiohttp

def wait_for_ollama(timeout=180):
    t0 = time.time()
    url = f"http://127.0.0.1:{OLLAMA_PORT}/api/tags"
    print(f"[Ollama] Waiting for service on port {OLLAMA_PORT}...")
    consecutive_success = 0
    while time.time() - t0 < timeout:
        try:
            resp = requests.get(url, timeout=5)
            if resp.ok:
                consecutive_success += 1
                print(f"[Ollama] Response OK ({consecutive_success}/3)")
                if consecutive_success >= 3:
                    print(f"✓ Ollama ready on port {OLLAMA_PORT}")
                    time.sleep(2)
                    return True
            else:
                consecutive_success = 0
        except:
            consecutive_success = 0
        time.sleep(3)
    return False

if not wait_for_ollama():
    print("Ollama log tail:\n", OLLAMA_LOG.read_text()[-2000:])
    raise RuntimeError("Ollama không khởi động được.")

print("✓ Ollama service is ready!")

# ---- Pull model (với retry) ----
_hdr(f"Pull model {MODEL}")
max_pull_attempts = 2  # Giảm số lần thử
for attempt in range(max_pull_attempts):
    try:
        print(f"[Pull] Attempt {attempt + 1}/{max_pull_attempts}...")
        result = run_cmd([str(OLLAMA_EXEC), "pull", MODEL], allow_fail=True)
        if result == 0:
            print(f"✓ Model {MODEL} pulled successfully")
            break
        else:
            print(f"✗ Pull failed with exit code {result}")
            if attempt < max_pull_attempts - 1:
                time.sleep(5)
    except Exception as e:
        print(f"✗ Pull error: {e}")
        if attempt < max_pull_attempts - 1:
            time.sleep(5)
        else:
            print(f"WARNING: Model pull failed")
            STATUS["last_err"] = f"Model pull failed: {e}"

# ---- Warm-up ----
def _warm(label, prompt):
    try:
        STATUS["phase"] = "warmup"
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": 300,
            "options": {
                "num_thread": 2,  # Giảm threads
                "num_ctx": 512,   # Giảm context
                "num_predict": 32  # Giảm predict
            }
        }
        r = requests.post(f"http://127.0.0.1:{OLLAMA_PORT}/api/chat", json=body, timeout=120)
        txt = r.json().get("message", {}).get("content", "") if r.ok else r.text[:120]
        print(f"{label}:", r.status_code, txt[:160])
        STATUS["phase"] = "ready"
    except Exception as e:
        STATUS["last_err"] = str(e)
        print(f"{label} lỗi:", e)

_warm("Warm-up(chat)", WARMUP_PROMPT)

# ---- Helper functions ----
def load_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except:
        return default

def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))

def brain_path(uid): return BRAIN_DIR / f"{uid}.json"
def brain_get(uid): return load_json(brain_path(uid), {"notes": []})
def brain_add(uid, text):
    d = brain_get(uid)
    d["notes"].append({"ts": int(time.time()), "text": text})
    save_json(brain_path(uid), d)

def brain_all_text(uid):
    notes = [n["text"] for n in brain_get(uid).get("notes", [])]
    s = "\n".join(f"- {t}" for t in notes[-3:]) if notes else "(chưa có)"  # Chỉ lấy 3 notes mới nhất
    if BRAIN_MAX_CHARS > 0 and len(s) > BRAIN_MAX_CHARS:
        s = s[:BRAIN_MAX_CHARS] + "…"
    return s

def hist_path(cid): return HIST_DIR / f"{cid}.json"
def hist_get(cid): return load_json(hist_path(cid), {"messages": []})
def hist_set(cid, msgs): save_json(hist_path(cid), {"messages": msgs})

def conf_get(): return load_json(CONF_FILE, {"auto_channels": [], "auto_train": True})
def conf_set(cfg): save_json(CONF_FILE, cfg)

# KB functions (SIMPLIFIED - không dùng scikit-learn)
def kb_append(item: dict):
    with open(KB_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

def kb_all():
    out = []
    if not KB_FILE.exists():
        return out
    for line in KB_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except:
            pass
    return out

def kb_recent(n=5):  # Giảm từ 10
    lines = KB_FILE.read_text(encoding="utf-8").splitlines() if KB_FILE.exists() else []
    items = []
    for ln in lines[-n:]:
        try:
            items.append(json.loads(ln))
        except:
            pass
    return items

def kb_clear():
    KB_FILE.write_text("")

def kb_compose_text(it):
    if it.get("type") == "qa":
        return f"Q: {it.get('q','').strip()}\nA: {it.get('a','').strip()}"
    return it.get("text", "").strip()

# RAG đơn giản - không dùng TF-IDF
def kb_inject_for_query(query: str, top_k=1, max_chars=200):
    items = kb_all()
    if not items:
        return None
    
    # Simple keyword matching
    qset = set(query.lower().split())
    scored = []
    for i, it in enumerate(items):
        text = kb_compose_text(it)
        tset = set(text.lower().split())
        score = len(qset & tset) / (len(qset) + 1e-6)
        if score > 0:
            scored.append((score, i, text))
    
    if not scored:
        return None
    
    scored.sort(reverse=True)
    lines = []
    for score, i, text in scored[:top_k]:
        snip = text.strip()[:max_chars]
        lines.append(f"- {snip}")
    
    return "Kiến thức:\n" + "\n".join(lines) if lines else None

def kb_inject_for_prompt(prompt: str):
    return kb_inject_for_query(prompt, top_k=RAG_TOPK, max_chars=RAG_CHARS)

# ---- Vision helpers (BỎ ảnh để tiết kiệm RAM) ----
IMG_MAX_BYTES = 500_000  # Giảm từ 1.5MB
IMG_MAX_ATTACH = 1  # CHỈ 1 ảnh

# ================= WEB SERVER (Flask) =================
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    uptime = int(time.time() - START_TIME)
    ram_mb = STATUS.get("ram_mb", 0)
    ram_pct = STATUS.get("ram_pct", 0)
    
    return jsonify({
        "status": "ok",
        "service": "Discord AI Bot (512MB optimized)",
        "model": MODEL,
        "phase": STATUS.get("phase", "ready"),
        "uptime_seconds": uptime,
        "ram": {
            "current_mb": round(ram_mb, 1),
            "limit_mb": RAM_LIMIT_MB,
            "usage_pct": round(ram_pct, 1),
            "status": STATUS.get("ram_status", "ok")
        },
        "ports": {"web": WEB_PORT, "ollama": OLLAMA_PORT}
    })

@app.route('/health')
def health():
    try:
        r = requests.get(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", timeout=5)
        ollama_ok = r.ok
    except:
        ollama_ok = False
    
    ram_mb = STATUS.get("ram_mb", 0)
    ram_pct = STATUS.get("ram_pct", 0)
    
    return jsonify({
        "bot_ready": bot.is_ready() if 'bot' in globals() else False,
        "ollama_ready": ollama_ok,
        "ram_ok": ram_mb < RAM_WARNING_MB,
        "status": STATUS,
        "ram_history": RAM_HISTORY[-20:] if RAM_HISTORY else []
    })

@app.route('/stats')
def stats():
    ram_mb = STATUS.get("ram_mb", 0)
    ram_pct = STATUS.get("ram_pct", 0)
    avg_ram = sum(h["mb"] for h in RAM_HISTORY[-10:]) / min(10, len(RAM_HISTORY)) if RAM_HISTORY else 0
    
    return jsonify({
        "model": MODEL,
        "phase": STATUS.get("phase"),
        "avg_sec": STATUS.get("avg_sec", 0.0),
        "total_requests": STATUS.get("count", 0),
        "active_slots": f"{len(ACTIVE_USERS)}/{MAX_ACTIVE_USERS}",
        "ram": {
            "current_mb": round(ram_mb, 1),
            "avg_mb": round(avg_ram, 1),
            "limit_mb": RAM_LIMIT_MB,
            "usage_pct": round(ram_pct, 1),
            "warning_mb": RAM_WARNING_MB,
            "critical_mb": RAM_CRITICAL_MB
        }
    })

@app.route('/ping')
def ping():
    return "pong", 200

@app.route('/ram')
def ram_info():
    """Endpoint chi tiết về RAM usage"""
    ram_mb = STATUS.get("ram_mb", 0)
    ram_pct = STATUS.get("ram_pct", 0)
    
    history_5m = [h for h in RAM_HISTORY if time.time() - h["ts"] < 300]
    history_1h = [h for h in RAM_HISTORY if time.time() - h["ts"] < 3600]
    
    return jsonify({
        "current": {
            "mb": round(ram_mb, 1),
            "pct": round(ram_pct, 1),
            "status": "🔴 CRITICAL" if ram_mb >= RAM_CRITICAL_MB else "🟡 WARNING" if ram_mb >= RAM_WARNING_MB else "🟢 OK"
        },
        "limits": {
            "total_mb": RAM_LIMIT_MB,
            "warning_mb": RAM_WARNING_MB,
            "critical_mb": RAM_CRITICAL_MB
        },
        "stats": {
            "avg_5m": round(sum(h["mb"] for h in history_5m) / len(history_5m), 1) if history_5m else 0,
            "max_5m": round(max((h["mb"] for h in history_5m), default=0), 1),
            "avg_1h": round(sum(h["mb"] for h in history_1h) / len(history_1h), 1) if history_1h else 0,
            "max_1h": round(max((h["mb"] for h in history_1h), default=0), 1)
        },
        "history_count": len(RAM_HISTORY),
        "last_cleanup": STATUS.get("last_cleanup", "never")
    })

def run_flask():
    try:
        print(f"[Flask] Starting on 0.0.0.0:{WEB_PORT}...")
        app.run(host='0.0.0.0', port=WEB_PORT, debug=False, use_reloader=False)
    except OSError as e:
        print(f"[Flask] Fatal error: {e}")
        raise

# ================= Discord bot =================
import discord
from discord.ext import commands

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)
allowed_mentions = discord.AllowedMentions(everyone=False, users=True, roles=False, replied_user=True)

session: aiohttp.ClientSession | None = None
START_TIME = time.time()
SEMA = asyncio.Semaphore(MAX_ACTIVE_USERS)
ACTIVE_USERS = set()
NOTIFIED = {}

DEFAULT_SYSTEM = "Bạn là trợ lý AI, trả lời ngắn gọn bằng tiếng Việt (≤1900 ký tự)."

# PROFILES tối ưu cho 512MB
ADAPTIVE_PROFILES = [
    {"name": "L3-ultra", "num_ctx": 512, "num_predict": 128, "num_batch": 16},  # Ultra minimal
    {"name": "L2-safe", "num_ctx": 768, "num_predict": 192, "num_batch": 24},
    {"name": "L1-fast", "num_ctx": 1024, "num_predict": 256, "num_batch": 32},
]

def _base_opts(profile):
    return {
        "num_thread": 2,  # Fixed 2 threads
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 30,
        "repeat_penalty": 1.1,
        "num_ctx": profile["num_ctx"],
        "num_predict": profile["num_predict"],
        "num_batch": profile["num_batch"]
    }

async def ensure_session():
    global session
    if session is None or session.closed:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQ_TIMEOUT))

def clip_one_msg(t):
    return t if len(t) <= DISCORD_MAX else t[:DISCORD_MAX - 12] + " …[rút gọn]"

def build_messages_with_brain(uid, msgs, extra_system=None):
    brain_txt = brain_all_text(uid)
    sys = DEFAULT_SYSTEM
    if brain_txt != "(chưa có)":
        sys += f"\nGhi nhớ: {brain_txt}"
    if extra_system:
        sys += "\n" + extra_system.strip()
    return [{"role": "system", "content": sys}] + msgs

def human_timedelta(seconds: float):
    seconds = int(seconds)
    h, seconds = divmod(seconds, 3600)
    m, s = divmod(seconds, 60)
    parts = []
    if h: parts.append(f"{h}h")
    if m: parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)

async def extract_images_from_message(msg: discord.Message):
    imgs = []
    for a in msg.attachments[:IMG_MAX_ATTACH]:
        ctype = (a.content_type or "").lower()
        if ctype.startswith("image/"):
            try:
                b = await a.read()
                if len(b) > IMG_MAX_BYTES:
                    b = b[:IMG_MAX_BYTES]
                imgs.append(base64.b64encode(b).decode("utf-8"))
            except:
                continue
    return imgs

async def _chat_once(messages, model, opts, stream=False):  # BẮT BUỘC stream=False để tiết kiệm RAM
    await ensure_session()
    url = f"http://127.0.0.1:{OLLAMA_PORT}/api/chat"
    body = {"model": model, "messages": messages, "stream": stream, "keep_alive": 300, "options": opts}
    
    async with session.post(url, json=body) as resp:
        txt = await resp.text()
        if resp.status >= 400:
            raise RuntimeError(f"CHAT HTTP {resp.status}")
        try:
            return json.loads(txt).get("message", {}).get("content", "") or "(rỗng)"
        except:
            return "(rỗng)"

async def ollama_chat(messages, model: str):
    last_error = None
    profiles = sorted(ADAPTIVE_PROFILES, key=lambda p: 0 if p["name"] == PREFERRED_PROFILE else 1)
    for prof in profiles:
        opts = _base_opts(prof)
        try:
            return await _chat_once(messages, model, opts, stream=False)
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"{type(last_error).__name__}: {last_error}")

async def can_start_or_notify(author, channel) -> bool:
    now = time.time()
    if author.id in ACTIVE_USERS:
        if now - NOTIFIED.get(("self", author.id), 0) >= 10:
            NOTIFIED[("self", author.id)] = now
            await channel.send(f"{author.mention} Đang xử lý tin trước, đợi xíu nhé.", allowed_mentions=allowed_mentions)
        return False
    if len(ACTIVE_USERS) >= MAX_ACTIVE_USERS:
        if now - NOTIFIED.get(("full", author.id), 0) >= 30:
            NOTIFIED[("full", author.id)] = now
            await channel.send(f"{author.mention} Đang bận, đợi ~20s nhé.", allowed_mentions=allowed_mentions)
        return False
    ACTIVE_USERS.add(author.id)
    return True

def release_user(author_id: int):
    ACTIVE_USERS.discard(author_id)

def build_user_message(prompt: str, images=None):
    msg = {"role": "user", "content": prompt}
    if images:
        msg["images"] = images
    return msg

async def ai_flow(channel, author, prompt: str, *, extra_system=None, mention_user=None, suppress_mention=False, images=None):
    can = await can_start_or_notify(author, channel)
    if not can:
        return
    STATUS["phase"] = "answering"
    t0 = time.time()

    h = hist_get(channel.id)
    msgs = h.get("messages", [])
    user_msg = build_user_message(prompt, images)
    msgs.append(user_msg)

    rag_extra = kb_inject_for_prompt(prompt)
    merged = build_messages_with_brain(
        author.id,
        msgs,
        extra_system=rag_extra if not extra_system else extra_system + "\n" + (rag_extra or "")
    )

    try:
        async with SEMA:
            reply = await ollama_chat(merged, MODEL)
    except Exception as e:
        STATUS["last_err"] = str(e)
        reply = f"Lỗi: {type(e).__name__}"
    finally:
        msgs.append({"role": "assistant", "content": reply})
        hist_set(channel.id, msgs[-HISTORY_TURNS * 2:])
        release_user(author.id)

    try:
        kb_append({
            "type": "qa",
            "q": prompt,
            "a": reply,
            "author_id": author.id,
            "ts": int(time.time())
        })
    except:
        pass

    dt = time.time() - t0
    avg_sec = (STATUS.get("avg_sec", 15.0) * STATUS.get("count", 0) + dt) / (STATUS.get("count", 0) + 1)
    STATUS["avg_sec"] = float(avg_sec)
    STATUS["count"] = STATUS.get("count", 0) + 1
    STATUS["phase"] = "ready"

    if suppress_mention or (hasattr(channel, "guild") and channel.guild is None):
        await channel.send(clip_one_msg(reply), allowed_mentions=discord.AllowedMentions.none())
    else:
        target = mention_user or author
        await channel.send(f"{target.mention} {clip_one_msg(reply)}", allowed_mentions=allowed_mentions)

# ===== Events =====
@bot.event
async def on_ready():
    STATUS["phase"] = "ready"
    try:
        if not getattr(bot, "_synced", False):
            await bot.tree.sync()
            bot._synced = True
    except:
        pass
    print(f"Bot online: {bot.user} (ID: {bot.user.id})")
    print(f"Web server: http://0.0.0.0:{WEB_PORT}")
    
    # Start RAM monitoring
    bot.loop.create_task(ram_monitor_loop())
    print("✓ RAM monitoring started")
    
    try:
        await bot.change_presence(
            activity=discord.Game(name=f"RAM: {STATUS.get('ram_mb', 0):.0f}/{RAM_LIMIT_MB}MB"),
            status=discord.Status.online
        )
    except:
        pass
    for g in bot.guilds:
        try:
            me = g.me
            if me and me.nick != BOT_NICKNAME:
                await me.edit(nick=BOT_NICKNAME)
        except:
            pass

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    content = message.content.strip()

    # DM
    if message.guild is None:
        if content and not content.startswith("!"):
            imgs = await extract_images_from_message(message)
            await ai_flow(message.channel, message.author, content, suppress_mention=True, images=imgs)
            return

    # Auto mode
    cfg = conf_get()
    auto_channels = set(cfg.get("auto_channels", []))
    if message.guild and message.channel.id in auto_channels and not content.startswith("!"):
        imgs = await extract_images_from_message(message)
        await ai_flow(message.channel, message.author, content, images=imgs)
        return

    # Mention bot
    if (bot.user in message.mentions) and not content.startswith("!"):
        clean = re.sub(r"<@!?\d+>", "", content).strip()
        if clean:
            imgs = await extract_images_from_message(message)
            await ai_flow(message.channel, message.author, clean, images=imgs)
        else:
            await message.reply("Bạn cần nhập nội dung sau khi mention.", allowed_mentions=allowed_mentions)
        return

    await bot.process_commands(message)

# ===== Commands =====
@bot.command(name="ai")
async def ai_cmd(ctx, *, prompt: str):
    imgs = await extract_images_from_message(ctx.message)
    await ai_flow(ctx.channel, ctx.author, prompt, images=imgs)

@bot.command(name="describe")
async def describe_cmd(ctx, *, hint: str = "Mô tả ảnh."):
    imgs = await extract_images_from_message(ctx.message)
    if not imgs:
        await ctx.reply("Vui lòng đính kèm ảnh.", allowed_mentions=allowed_mentions)
        return
    await ai_flow(ctx.channel, ctx.author, hint, images=imgs)

@bot.command(name="train")
async def train_cmd(ctx, sub: str = "status", *, payload: str = ""):
    ssub = sub.lower() if sub else "status"
    if ssub in ("status", "stat"):
        await ctx.reply(
            f"{ctx.author.mention} KB: {len(kb_all())} mục.",
            allowed_mentions=allowed_mentions
        )
    elif ssub in ("list", "ls"):
        items = kb_recent(5)
        if not items:
            await ctx.reply(f"{ctx.author.mention} KB trống.", allowed_mentions=allowed_mentions)
            return
        lines = []
        for it in items:
            if it.get("type") == "qa":
                s = f"Q={it.get('q','')[:50]}"
            else:
                s = f"{it.get('text','')[:100]}"
            lines.append(s)
        await ctx.reply(
            clip_one_msg("KB:\n" + "\n".join(f"- {l}" for l in lines)),
            allowed_mentions=allowed_mentions
        )
    elif ssub == "clear":
        if not (ctx.author.guild_permissions.manage_messages if ctx.guild else False):
            await ctx.reply(
                f"{ctx.author.mention} Cần quyền Manage Messages.",
                allowed_mentions=allowed_mentions
            )
            return
        kb_clear()
        await ctx.reply(f"{ctx.author.mention} Đã xóa KB.", allowed_mentions=allowed_mentions)
    else:
        await ctx.reply(
            f"{ctx.author.mention} Dùng: !train status | !train list | !train clear",
            allowed_mentions=allowed_mentions
        )

@bot.command(name="auto")
async def auto_cmd(ctx, mode: str = None):
    cfg = conf_get()
    s = set(cfg.get("auto_channels", []))
    if mode is None:
        await ctx.reply(
            f"{ctx.author.mention} Auto: {'ON' if ctx.channel.id in s else 'OFF'}",
            allowed_mentions=allowed_mentions
        )
        return
    if mode.lower() in ("on", "true", "1"):
        s.add(ctx.channel.id)
        msg = "đã bật"
    elif mode.lower() in ("off", "false", "0"):
        s.discard(ctx.channel.id)
        msg = "đã tắt"
    else:
        await ctx.reply(
            f"{ctx.author.mention} Dùng: !auto on | !auto off",
            allowed_mentions=allowed_mentions
        )
        return
    cfg["auto_channels"] = list(s)
    conf_set(cfg)
    await ctx.reply(f"{ctx.author.mention} Auto {msg}.", allowed_mentions=allowed_mentions)

@bot.command(name="remember")
async def remember_cmd(ctx, *, text: str):
    brain_add(ctx.author.id, text.strip())
    await ctx.reply(f"{ctx.author.mention} Đã lưu.", allowed_mentions=allowed_mentions)

@bot.command(name="brain")
async def brain_cmd(ctx, action: str = "show", who: str = ""):
    target = ctx.author
    if who and ctx.message.mentions:
        target = ctx.message.mentions[0]
    
    if action == "show":
        await ctx.reply(
            f"{ctx.author.mention} Brain của {target.mention}:\n{clip_one_msg(brain_all_text(target.id))}",
            allowed_mentions=allowed_mentions
        )
    elif action == "clear":
        save_json(brain_path(target.id), {"notes": []})
        await ctx.reply(
            f"{ctx.author.mention} Đã xóa brain.",
            allowed_mentions=allowed_mentions
        )
    else:
        await ctx.reply(
            f"{ctx.author.mention} Dùng: !brain show | !brain clear",
            allowed_mentions=allowed_mentions
        )

@bot.command(name="clear")
async def clear_cmd(ctx):
    hist_set(ctx.channel.id, [])
    await ctx.reply(
        f"{ctx.author.mention} Đã xóa lịch sử.",
        allowed_mentions=allowed_mentions
    )

@bot.command(name="model")
async def model_cmd(ctx, *, m: str = ""):
    global MODEL
    if m:
        MODEL = m.strip()
        STATUS["model"] = MODEL
        await ctx.reply(f"{ctx.author.mention} Model: {MODEL}", allowed_mentions=allowed_mentions)
    else:
        await ctx.reply(f"{ctx.author.mention} Model: {MODEL}", allowed_mentions=allowed_mentions)

@bot.command(name="info")
async def info_cmd(ctx):
    ws = int(bot.latency * 1000)
    up = int(time.time() - START_TIME)
    await ctx.reply(
        f"{ctx.author.mention}\nModel: {MODEL}\n"
        f"Uptime: {human_timedelta(up)}\nWS Ping: {ws}ms\n"
        f"RAM Limit: 512MB\nWeb: http://0.0.0.0:{WEB_PORT}",
        allowed_mentions=allowed_mentions
    )

@bot.command(name="status")
async def status_cmd(ctx):
    ph = STATUS.get("phase", "ready")
    mdl = STATUS.get("model", MODEL)
    avg = STATUS.get("avg_sec", 0.0)
    err = STATUS.get("last_err", "")

    msg = f"Phase: {ph}\nModel: {mdl}\nAvg: ~{avg:.1f}s"
    if err:
        msg += f"\nError: {err[:100]}"
    await ctx.reply(msg, allowed_mentions=allowed_mentions)

@bot.command(name="stats")
async def stats_cmd(ctx):
    ram_mb = STATUS.get("ram_mb", 0)
    ram_pct = STATUS.get("ram_pct", 0)
    ram_status = "🔴" if ram_mb >= RAM_CRITICAL_MB else "🟡" if ram_mb >= RAM_WARNING_MB else "🟢"
    
    slots = f"{len(ACTIVE_USERS)}/{MAX_ACTIVE_USERS}"
    msg = (
        f"Model: {MODEL}\nPhase: {STATUS.get('phase','ready')}\n"
        f"Avg: ~{STATUS.get('avg_sec',0.0):.1f}s\n"
        f"Slots: {slots}\n"
        f"{ram_status} RAM: {ram_mb:.1f}/{RAM_LIMIT_MB}MB ({ram_pct:.0f}%)\n"
        f"Web: http://0.0.0.0:{WEB_PORT}"
    )
    await ctx.reply(msg, allowed_mentions=allowed_mentions)

@bot.command(name="pull")
async def pull_cmd(ctx, *, model: str):
    await ensure_session()
    STATUS["phase"] = "pulling"
    STATUS["model"] = model
    msg = await ctx.reply(f"{ctx.author.mention} Pull {model}: 0%", allowed_mentions=allowed_mentions)
    
    try:
        async with session.post(
            f"http://127.0.0.1:{OLLAMA_PORT}/api/pull",
            json={"name": model, "stream": True}
        ) as resp:
            last_pct = -1
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    comp = int(data.get("completed") or 0)
                    tot = int(data.get("total") or 1)
                    pct = int(comp * 100 / tot) if tot else 0
                    
                    if pct != last_pct and pct % 10 == 0:
                        last_pct = pct
                        await msg.edit(content=f"{ctx.author.mention} Pull: {pct}%")
                except:
                    continue
        
        STATUS["phase"] = "ready"
        await msg.edit(content=f"{ctx.author.mention} Pull done ✅")
    except Exception as e:
        STATUS["last_err"] = str(e)
        await msg.edit(content=f"{ctx.author.mention} Pull lỗi: {e}")

@bot.command(name="warm")
async def warm_cmd(ctx):
    try:
        STATUS["phase"] = "warmup"
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "options": {"num_thread": 2, "num_ctx": 512, "num_predict": 32}
        }
        r = requests.post(f"http://127.0.0.1:{OLLAMA_PORT}/api/chat", json=body, timeout=120)
        STATUS["phase"] = "ready"
        await ctx.reply(f"{ctx.author.mention} Warm: {r.status_code}", allowed_mentions=allowed_mentions)
    except Exception as e:
        await ctx.reply(f"{ctx.author.mention} Warm lỗi: {e}", allowed_mentions=allowed_mentions)

@bot.command(name="health")
async def health_cmd(ctx):
    try:
        r = requests.get(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", timeout=5)
        await ctx.reply(f"{ctx.author.mention} Health: {r.status_code}", allowed_mentions=allowed_mentions)
    except Exception as e:
        await ctx.reply(f"{ctx.author.mention} Health lỗi: {e}", allowed_mentions=allowed_mentions)

@bot.command(name="help")
async def help_cmd(ctx):
    text = (
        "**Commands**\n"
        "- @mention bot | !ai <text> | DM bot\n"
        "- !describe (kèm ảnh)\n"
        "- !train status/list/clear\n"
        "- !auto on/off | !remember <text>\n"
        "- !brain show/clear | !clear\n"
        "- !status | !stats | !info | !health\n"
        "- !model [id] | !pull <model> | !warm\n\n"
        f"Model: {MODEL} | RAM: 512MB\n"
        f"Web: http://0.0.0.0:{WEB_PORT}"
    )
    try:
        await ctx.reply(
            embed=discord.Embed(title="Help", description=text, color=0x5865F2),
            allowed_mentions=allowed_mentions
        )
    except:
        await ctx.reply(text, allowed_mentions=allowed_mentions)

# ======== Main ========
async def main():
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print(f"✓ Flask started on port {WEB_PORT}")
    
    TOKEN = os.environ.get("DISCORD_TOKEN")
    if not TOKEN:
        raise RuntimeError("DISCORD_TOKEN not set!")
    
    print("Starting Discord bot...")
    print(f"Web: http://0.0.0.0:{WEB_PORT}")
    print(f"Ollama: http://127.0.0.1:{OLLAMA_PORT}")
    await bot.start(TOKEN)

if __name__ == "__main__":
    try:
        TOKEN = os.environ.get("DISCORD_TOKEN")
        if not TOKEN:
            print("ERROR: DISCORD_TOKEN not set!")
            sys.exit(1)
        
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✓ Bot stopped")
    except Exception as e:
        print(f"Fatal: {e}")
        traceback.print_exc()
        sys.exit(1)
