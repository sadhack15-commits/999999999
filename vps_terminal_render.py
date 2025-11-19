# ==========================
# Discord bot + Ollama (NO SUDO, với Web Server)
# Model: gemma3:4b (Text + Image)
# ==========================
import os, sys, subprocess, time, asyncio, json, re, traceback, zipfile, io, base64
from pathlib import Path
from threading import Thread

# ---- Quick setup (pip + ollama) ----
def pip_install(pkgs):
    print("[pip] Installing:", " ".join(pkgs))
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", *pkgs], check=True)

pip_install([
    "discord.py>=2.4.0",
    "aiohttp>=3.9.5",
    "requests>=2.31.0",
    "scikit-learn>=1.3.0",
    "nest_asyncio>=1.6.0",
    "flask>=3.0.0"
])

import nest_asyncio
import requests  # Import sớm để dùng cho Ollama install
nest_asyncio.apply()

# ---- Cài Ollama (NO SUDO - user mode) ----
print("[ollama] Installing in user mode...")

OLLAMA_DIR = Path.home() / ".ollama"
OLLAMA_BIN = OLLAMA_DIR / "bin"
OLLAMA_EXEC = OLLAMA_BIN / "ollama"

def install_ollama_user():
    """Cài Ollama không cần sudo - Download từ GitHub releases"""
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
    
    # GitHub releases có file .tgz (tarball)
    versions_to_try = ["v0.6.5", "v0.5.14", "v0.5.10", "v0.5.8", "v0.5.0", "v0.1.30"]
    
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
            
            if response.status_code == 404:
                print(f"  ✗ Not found (404)")
                continue
            
            if response.status_code != 200:
                print(f"  ✗ HTTP {response.status_code}")
                continue
            
            total_size = int(response.headers.get('content-length', 0))
            if total_size == 0:
                print(f"  ✗ Empty or unknown size")
                continue
            
            print(f"  ✓ Found! Downloading {total_size/1048576:.1f}MB...")
            
            # Download to temp file
            tgz_path = OLLAMA_BIN / filename
            downloaded = 0
            last_print = 0
            
            with open(tgz_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Print every 5MB
                        if downloaded - last_print >= 5*1024*1024:
                            pct = (downloaded / total_size) * 100
                            print(f"    {pct:.0f}% ({downloaded/1048576:.0f}MB/{total_size/1048576:.0f}MB)")
                            last_print = downloaded
            
            print(f"  ✓ Downloaded: {tgz_path.stat().st_size/1048576:.1f}MB")
            
            # Extract tarball
            print(f"  Extracting...")
            try:
                with tarfile.open(tgz_path, 'r:gz') as tar:
                    # Tìm binary trong tarball
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
                    
                    # Extract binary
                    tar.extract(ollama_member, path=OLLAMA_BIN)
                    extracted_path = OLLAMA_BIN / ollama_member.name
                    
                    # Move to final location
                    if extracted_path != OLLAMA_EXEC:
                        extracted_path.rename(OLLAMA_EXEC)
                    
                    OLLAMA_EXEC.chmod(0o755)
                
                # Cleanup
                tgz_path.unlink()
                
                # Verify
                file_size = OLLAMA_EXEC.stat().st_size
                if file_size < 10*1024*1024:  # Less than 10MB is suspicious
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
            
        except requests.exceptions.Timeout:
            print(f"  ✗ Download timeout")
            last_error = "Timeout"
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Request error: {e}")
            last_error = str(e)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            last_error = str(e)
            traceback.print_exc()
    
    raise RuntimeError(f"Failed to install Ollama from all sources. Last error: {last_error}")

install_ollama_user()

# Thêm vào PATH
os.environ["PATH"] = f"{OLLAMA_BIN}:{os.environ['PATH']}"

# ---- Config mặc định ----
MODEL            = "gemma3:4b"
OLLAMA_PORT      = 11434
WEB_PORT         = int(os.environ.get("PORT", 8080))  # Render sẽ set PORT
WARMUP_PROMPT    = "ping"
BOT_NICKNAME     = "Victory_vn_AI_✨"
REQ_TIMEOUT      = 300
DISCORD_MAX      = 1900
HISTORY_TURNS    = 4
MAX_ACTIVE_USERS = 2
BRAIN_MAX_CHARS  = 0
RAG_TOPK         = 2
RAG_CHARS        = 400
PREFERRED_PROFILE= "L1-fast"

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
    "uptime": 0
}

def _hdr(title): print("\n" + "="*80 + f"\n[ {title} ]\n" + "="*80)

# ---- Data dirs ----
DATA_DIR   = Path.home() / "bot_data"  # Dùng home directory
BRAIN_DIR  = DATA_DIR / "brain" / "users"
HIST_DIR   = DATA_DIR / "history" / "channels"
CONF_FILE  = DATA_DIR / "config.json"
KB_FILE    = DATA_DIR / "kb.jsonl"
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
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env=env)
    for line in p.stdout:
        print(line, end="")
    ret = p.wait()
    if ret != 0 and not allow_fail:
        raise RuntimeError(f"Command failed (exit={ret}): {' '.join(cmd)}")
    return ret

run_cmd(["pkill", "-f", "ollama serve"], "Kill tiến trình ollama cũ", allow_fail=True)

# ---- Start Ollama ----
_hdr("Start Ollama serve (user mode)")
OLLAMA_LOG.write_text("")
ollama_env = os.environ.copy()
ollama_env["OLLAMA_HOST"] = f"127.0.0.1:{OLLAMA_PORT}"
ollama_env["OLLAMA_MODELS"] = str(DATA_DIR / "models")

subprocess.Popen(
    [str(OLLAMA_EXEC), "serve"],
    stdout=open(OLLAMA_LOG, 'w'),
    stderr=subprocess.STDOUT,
    env=ollama_env
)
print(f"Ollama đang khởi động... (log: {OLLAMA_LOG})")

import requests, aiohttp

def wait_for_ollama(timeout=180):
    t0 = time.time()
    url = f"http://127.0.0.1:{OLLAMA_PORT}/api/tags"
    while time.time() - t0 < timeout:
        try:
            if requests.get(url, timeout=3).ok:
                return True
        except:
            pass
        time.sleep(1)
    return False

if not wait_for_ollama():
    print("Ollama log tail:\n", OLLAMA_LOG.read_text()[-2000:])
    raise RuntimeError("Ollama không khởi động được.")

print("✓ Ollama service is ready!")

# ---- Pull model ----
_hdr(f"Pull model {MODEL}")
run_cmd([str(OLLAMA_EXEC), "pull", MODEL], allow_fail=False)

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
                "num_thread": (os.cpu_count() or 2),
                "num_ctx": 1536,
                "num_predict": 64
            }
        }
        r = requests.post(f"http://127.0.0.1:{OLLAMA_PORT}/api/chat", json=body, timeout=120)
        try:
            txt = r.json().get("message", {}).get("content", "")
        except:
            txt = r.text[:120]
        print(f"{label}:", r.status_code, txt[:160])
        STATUS["phase"] = "ready"
    except Exception as e:
        STATUS["last_err"] = str(e)
        print(f"{label} lỗi:", e)

_warm("Warm-up(chat)", WARMUP_PROMPT)

# ---- Helper functions (Brain, History, KB, RAG...) ----
def load_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except:
        return default

def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))

def brain_path(uid): return BRAIN_DIR / f"{uid}.json"
def brain_get(uid):  return load_json(brain_path(uid), {"notes": []})
def brain_add(uid, text):
    d = brain_get(uid)
    d["notes"].append({"ts": int(time.time()), "text": text})
    save_json(brain_path(uid), d)

def brain_all_text(uid):
    notes = [n["text"] for n in brain_get(uid).get("notes", [])]
    s = "\n".join(f"- {t}" for t in notes) if notes else "(chưa có ghi nhớ)"
    if BRAIN_MAX_CHARS > 0 and len(s) > BRAIN_MAX_CHARS:
        s = s[:BRAIN_MAX_CHARS] + "…"
    return s

def hist_path(cid): return HIST_DIR / f"{cid}.json"
def hist_get(cid):  return load_json(hist_path(cid), {"messages": []})
def hist_set(cid, msgs): save_json(hist_path(cid), {"messages": msgs})

def conf_get(): return load_json(CONF_FILE, {"auto_channels": [], "auto_train": True})
def conf_set(cfg): save_json(CONF_FILE, cfg)

# KB functions
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

def kb_recent(n=10):
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

def kb_inject_for_query(query: str, top_k=RAG_TOPK, max_chars=RAG_CHARS):
    items = kb_all()
    if not items:
        return None
    texts = [kb_compose_text(it) for it in items]
    idxs = []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
        X = vec.fit_transform(texts + [query])
        sims = cosine_similarity(X[-1], X[:-1]).ravel()
        idxs = sims.argsort()[::-1][:top_k].tolist()
    except Exception:
        qset = set(query.lower().split())
        scored = []
        for i, t in enumerate(texts):
            sset = set(t.lower().split())
            score = len(qset & sset) / (len(qset) + 1e-6)
            scored.append((score, i))
        idxs = [i for (s, i) in sorted(scored, reverse=True)[:top_k] if s > 0]
    if not idxs:
        return None
    lines = []
    for i in idxs:
        if i is None or i >= len(texts):
            continue
        snip = texts[i].strip()
        if max_chars and len(snip) > max_chars:
            snip = snip[:max_chars] + "…"
        lines.append(f"- {snip}")
    return "Kiến thức liên quan:\n" + "\n".join(lines) if lines else None

def kb_inject_for_prompt(prompt: str):
    return kb_inject_for_query(prompt, top_k=RAG_TOPK, max_chars=RAG_CHARS)

# ---- Vision helpers ----
IMG_MAX_BYTES = 1_500_000
IMG_MAX_ATTACH = 3

# ================= WEB SERVER (Flask) =================
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def home():
    """Health check endpoint cho UptimeRobot"""
    uptime = int(time.time() - START_TIME)
    return jsonify({
        "status": "ok",
        "service": "Discord AI Bot",
        "model": STATUS.get("model", MODEL),
        "phase": STATUS.get("phase", "ready"),
        "uptime_seconds": uptime,
        "uptime_human": human_timedelta(uptime),
        "active_users": len(ACTIVE_USERS),
        "max_users": MAX_ACTIVE_USERS,
        "avg_response_time": f"{STATUS.get('avg_sec', 0.0):.2f}s",
        "kb_size": len(kb_all())
    })

@app.route('/health')
def health():
    """Endpoint health check chi tiết"""
    try:
        r = requests.get(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", timeout=5)
        ollama_ok = r.ok
    except:
        ollama_ok = False
    
    return jsonify({
        "bot_ready": bot.is_ready() if 'bot' in globals() else False,
        "ollama_ready": ollama_ok,
        "status": STATUS
    })

@app.route('/stats')
def stats():
    """Statistics endpoint"""
    return jsonify({
        "model": STATUS.get("model", MODEL),
        "phase": STATUS.get("phase", "ready"),
        "avg_sec": STATUS.get("avg_sec", 0.0),
        "total_requests": STATUS.get("count", 0),
        "active_slots": f"{len(ACTIVE_USERS)}/{MAX_ACTIVE_USERS}",
        "kb_entries": len(kb_all()),
        "last_error": STATUS.get("last_err", "")
    })

@app.route('/ping')
def ping():
    """Simple ping endpoint"""
    return "pong", 200

def run_flask():
    """Chạy Flask server trong thread riêng"""
    app.run(host='0.0.0.0', port=WEB_PORT, debug=False, use_reloader=False)

# ================= Discord bot =================
import discord
from discord.ext import commands
from discord import app_commands

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=commands.when_mentioned_or("!"), intents=intents, help_command=None)
allowed_mentions = discord.AllowedMentions(everyone=False, users=True, roles=False, replied_user=True)

session: aiohttp.ClientSession | None = None
START_TIME = time.time()
SEMA = asyncio.Semaphore(MAX_ACTIVE_USERS)
ACTIVE_USERS = set()
NOTIFIED = {}
AUTO_TRAIN_ENABLED = True

DEFAULT_SYSTEM = (
    "Bạn là trợ lý AI nói tiếng Việt, lịch sự, súc tích. "
    "Luôn trả lời ngắn gọn, rõ ràng, chỉ 1 tin nhắn (≤ ~1900 ký tự). "
    "Nếu không chắc, hãy nói 'Mình không chắc' và gợi ý cách kiểm chứng."
)

ADAPTIVE_PROFILES = [
    {"name": "L1-fast", "num_ctx": 3072, "num_predict": 256, "num_batch": 64},
    {"name": "L2-safe", "num_ctx": 2048, "num_predict": 224, "num_batch": 48},
    {"name": "L3-ultra", "num_ctx": 1536, "num_predict": 192, "num_batch": 32},
]

def _base_opts(profile):
    return {
        "num_thread": (os.cpu_count() or 2),
        "temperature": 0.25,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.15,
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
    sys = DEFAULT_SYSTEM + f"\nGhi nhớ về người dùng:\n{brain_txt}\n"
    if extra_system:
        sys += "\n" + extra_system.strip()
    return [{"role": "system", "content": sys}] + msgs

def human_timedelta(seconds: float):
    seconds = int(seconds)
    d, seconds = divmod(seconds, 86400)
    h, seconds = divmod(seconds, 3600)
    m, s = divmod(seconds, 60)
    parts = []
    if d: parts.append(f"{d}d")
    if h: parts.append(f"{h}h")
    if m: parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)

async def extract_images_from_message(msg: discord.Message):
    imgs = []
    for a in msg.attachments[:IMG_MAX_ATTACH]:
        ctype = (a.content_type or "").lower()
        fname = a.filename or ""
        if (ctype.startswith("image/") or fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))):
            try:
                b = await a.read()
                if len(b) > IMG_MAX_BYTES:
                    b = b[:IMG_MAX_BYTES]
                imgs.append(base64.b64encode(b).decode("utf-8"))
            except Exception as e:
                STATUS["last_err"] = f"read_image_error: {e}"
                continue
    return imgs

async def _chat_once(messages, model, opts, stream=True):
    await ensure_session()
    url = f"http://127.0.0.1:{OLLAMA_PORT}/api/chat"
    body = {"model": model, "messages": messages, "stream": stream, "keep_alive": 300, "options": opts}
    if stream:
        async with session.post(url, json=body) as resp:
            acc = ""
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except:
                    continue
                msg = data.get("message", {})
                if "content" in msg:
                    acc += msg["content"]
                if data.get("done"):
                    break
            if resp.status >= 400 and not acc:
                raise RuntimeError(f"CHAT HTTP {resp.status}")
            return acc or "(rỗng)"
    else:
        async with session.post(url, json=body) as resp:
            txt = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"CHAT HTTP {resp.status}: {txt[:300]}")
            try:
                return json.loads(txt).get("message", {}).get("content", "") or "(rỗng)"
            except:
                return "(rỗng)"

async def _generate_fallback(messages, model, opts):
    await ensure_session()
    url = f"http://127.0.0.1:{OLLAMA_PORT}/api/generate"
    sys_txt = ""
    seq = []
    for m in messages:
        r = m.get("role")
        c = m.get("content", "").strip()
        if r == "system":
            sys_txt = c
        elif r == "user":
            seq.append(f"User: {c}")
        elif r == "assistant":
            seq.append(f"Assistant: {c}")
    full_prompt = ((f"System: {sys_txt}\n") if sys_txt else "") + "\n".join(seq) + "\nAssistant:"
    body = {"model": model, "prompt": full_prompt, "stream": False, "keep_alive": 300, "options": opts}
    async with session.post(url, json=body) as resp:
        txt = await resp.text()
        if resp.status >= 400:
            raise RuntimeError(f"GEN HTTP {resp.status}: {txt[:300]}")
        try:
            return json.loads(txt).get("response", "") or "(rỗng)"
        except:
            out = ""
            for line in txt.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out += json.loads(line).get("response", "")
                except:
                    pass
            return out or "(rỗng)"

async def ollama_chat(messages, model: str):
    last_error = None
    profiles = sorted(ADAPTIVE_PROFILES, key=lambda p: 0 if p["name"] == PREFERRED_PROFILE else 1)
    for prof in profiles:
        opts = _base_opts(prof)
        try:
            return await _chat_once(messages, model, opts, stream=True)
        except Exception:
            try:
                return await _generate_fallback(messages, model, opts)
            except Exception as e2:
                last_error = e2
                continue
    raise RuntimeError(f"{type(last_error).__name__}: {last_error}")

async def can_start_or_notify(author, channel) -> bool:
    now = time.time()
    if author.id in ACTIVE_USERS:
        if now - NOTIFIED.get(("self", author.id), 0) >= 10:
            NOTIFIED[("self", author.id)] = now
            await channel.send(f"{author.mention} Đang xử lý tin trước, đợi xíu nhé. ❤️", allowed_mentions=allowed_mentions)
        return False
    if len(ACTIVE_USERS) >= MAX_ACTIVE_USERS:
        eta = max(8, int(STATUS.get("avg_sec", 15))) if STATUS.get("avg_sec", 0) > 0 else 20
        if now - NOTIFIED.get(("full", author.id), 0) >= 30:
            NOTIFIED[("full", author.id)] = now
            await channel.send(f"{author.mention} Mình đang bận tối đa {MAX_ACTIVE_USERS} người, ước ~{eta}s nữa nhé. ❤️", allowed_mentions=allowed_mentions)
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
        traceback.print_exc()
        reply = f"Lỗi LOCAL API: {type(e).__name__}: {e}"
    finally:
        msgs.append({"role": "assistant", "content": reply})
        hist_set(channel.id, msgs[-HISTORY_TURNS * 2:])
        release_user(author.id)

    try:
        if True:
            kb_append({
                "type": "qa",
                "q": prompt,
                "a": reply,
                "author_id": author.id,
                "channel_id": getattr(channel, 'id', None),
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

# Bot-to-bot logic
B2B_SESSION = {}

def b2b_start_for_initiator(cid, pid):
    B2B_SESSION[cid] = {"partner_id": pid, "allow_reply": False, "expires": time.time() + 60}

def b2b_allow_once_for_responder(cid, pid):
    B2B_SESSION[cid] = {"partner_id": pid, "allow_reply": True, "expires": time.time() + 30}

def b2b_can_reply(cid, aid):
    s = B2B_SESSION.get(cid)
    if not s:
        return False
    if time.time() > s["expires"]:
        B2B_SESSION.pop(cid, None)
        return False
    return s["allow_reply"] and s["partner_id"] == aid

def b2b_after_reply(cid):
    s = B2B_SESSION.get(cid)
    if not s:
        return
    s["allow_reply"] = False
    s["expires"] = time.time() + 5

# ===== Events =====
@bot.event
async def on_ready():
    STATUS["phase"] = "ready"
    try:
        if not getattr(bot, "_synced", False):
            await bot.tree.sync()
            bot._synced = True
    except Exception as e:
        print("Sync slash lỗi:", e)
    print(f"Bot online: {bot.user} (ID: {bot.user.id})")
    print(f"Web server: http://0.0.0.0:{WEB_PORT}")
    try:
        await bot.change_presence(
            activity=discord.Game(name=f"Web: 0.0.0.0:{WEB_PORT} | Nhắn @{BOT_NICKNAME}"),
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
        if bot.user in message.mentions:
            ch_id = message.channel.id
            author_id = message.author.id
            if ch_id not in B2B_SESSION:
                b2b_allow_once_for_responder(ch_id, author_id)
            if b2b_can_reply(ch_id, author_id):
                try:
                    await message.channel.send(
                        f"{message.author.mention} Xin chào người dùng! Tôi có thể giúp gì cho bạn hôm nay?",
                        allowed_mentions=allowed_mentions
                    )
                finally:
                    b2b_after_reply(ch_id)
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
            await message.reply("Bạn cần nhập nội dung sau khi mention mình nha.", allowed_mentions=allowed_mentions)
        return

    await bot.process_commands(message)

# ===== Commands =====
@bot.command(name="ai")
async def ai_cmd(ctx, *, prompt: str):
    imgs = await extract_images_from_message(ctx.message)
    await ai_flow(ctx.channel, ctx.author, prompt, images=imgs)

@bot.command(name="describe")
async def describe_cmd(ctx, *, hint: str = "Mô tả chi tiết nội dung ảnh."):
    imgs = await extract_images_from_message(ctx.message)
    if not imgs:
        await ctx.reply("Vui lòng đính kèm ít nhất 1 ảnh.", allowed_mentions=allowed_mentions)
        return
    await ai_flow(ctx.channel, ctx.author, hint, images=imgs)

@bot.command(name="chatai")
async def chatai_cmd(ctx):
    if not ctx.message.mentions:
        await ctx.reply(f"{ctx.author.mention} Dùng: !chatai @user", allowed_mentions=allowed_mentions)
        return
    target = ctx.message.mentions[0]
    can = await can_start_or_notify(ctx.author, ctx.channel)
    if not can:
        return
    try:
        async with SEMA:
            b2b_start_for_initiator(ctx.channel.id, target.id)
            await ctx.channel.send(
                f"{target.mention} Chào bạn! Tôi có thể giúp gì cho bạn hôm nay?",
                allowed_mentions=allowed_mentions
            )
    finally:
        release_user(ctx.author.id)

@bot.command(name="train")
async def train_cmd(ctx, sub: str = "status", *, payload: str = ""):
    ssub = sub.lower() if sub else "status"
    if ssub in ("status", "stat"):
        await ctx.reply(
            f"{ctx.author.mention} Auto-train: ON | KB: {len(kb_all())} mục.",
            allowed_mentions=allowed_mentions
        )
    elif ssub in ("list", "ls"):
        n = 10
        if payload.strip().isdigit():
            n = max(1, min(100, int(payload.strip())))
        items = kb_recent(n)
        if not items:
            await ctx.reply(f"{ctx.author.mention} KB trống.", allowed_mentions=allowed_mentions)
            return
        lines = []
        for it in items:
            if it.get("type") == "qa":
                s = f"QA: Q={it.get('q','')[:70]} | A={it.get('a','')[:70]}"
            else:
                s = f"DOC: {it.get('text','')[:140]}"
            lines.append(s)
        await ctx.reply(
            clip_one_msg("KB gần đây:\n" + "\n".join(f"- {l}" for l in lines)),
            allowed_mentions=allowed_mentions
        )
    elif ssub == "clear":
        if not (ctx.author.guild_permissions.manage_messages if ctx.guild else False):
            await ctx.reply(
                f"{ctx.author.mention} Cần quyền Manage Messages để xóa KB.",
                allowed_mentions=allowed_mentions
            )
            return
        kb_clear()
        await ctx.reply(f"{ctx.author.mention} Đã xóa toàn bộ KB.", allowed_mentions=allowed_mentions)
    else:
        await ctx.reply(
            f"{ctx.author.mention} Dùng: !train status | !train list [n] | !train clear",
            allowed_mentions=allowed_mentions
        )

@bot.command(name="auto")
async def auto_cmd(ctx, mode: str = None):
    cfg = conf_get()
    s = set(cfg.get("auto_channels", []))
    if mode is None:
        await ctx.reply(
            f"{ctx.author.mention} Auto ở kênh này đang: {'ON' if ctx.channel.id in s else 'OFF'}",
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
    await ctx.reply(f"{ctx.author.mention} Auto {msg} cho kênh này.", allowed_mentions=allowed_mentions)

@bot.command(name="remember")
async def remember_cmd(ctx, *, text: str):
    brain_add(ctx.author.id, text.strip())
    await ctx.reply(f"{ctx.author.mention} Đã lưu vào brain của bạn.", allowed_mentions=allowed_mentions)

@bot.command(name="brain")
async def brain_cmd(ctx, action: str = "show", who: str = ""):
    target = ctx.author
    if who:
        if ctx.message.mentions:
            target = ctx.message.mentions[0]
        else:
            try:
                target = await bot.fetch_user(int(who))
            except:
                pass
    if action == "show":
        await ctx.reply(
            f"{ctx.author.mention} Brain của {target.mention}:\n{clip_one_msg(brain_all_text(target.id))}",
            allowed_mentions=allowed_mentions
        )
    elif action == "clear":
        save_json(brain_path(target.id), {"notes": []})
        await ctx.reply(
            f"{ctx.author.mention} Đã xóa brain của {target.mention}.",
            allowed_mentions=allowed_mentions
        )
    else:
        await ctx.reply(
            f"{ctx.author.mention} Dùng: !brain show [@user] | !brain clear [@user]",
            allowed_mentions=allowed_mentions
        )

@bot.command(name="clear")
async def clear_cmd(ctx):
    hist_set(ctx.channel.id, [])
    await ctx.reply(
        f"{ctx.author.mention} Đã xóa lịch sử hội thoại kênh này.",
        allowed_mentions=allowed_mentions
    )

@bot.command(name="model")
async def model_cmd(ctx, *, m: str = ""):
    global MODEL
    if m:
        MODEL = m.strip()
        STATUS["model"] = MODEL
        await ctx.reply(f"{ctx.author.mention} Đã set model: {MODEL}", allowed_mentions=allowed_mentions)
    else:
        await ctx.reply(f"{ctx.author.mention} Model hiện tại: {MODEL}", allowed_mentions=allowed_mentions)

@bot.command(name="info")
async def info_cmd(ctx):
    ws = int(bot.latency * 1000)
    up = int(time.time() - START_TIME)
    await ctx.reply(
        f"{ctx.author.mention}\nCreator: victory_vn\nModel: {MODEL}\n"
        f"Uptime: {human_timedelta(up)}\nWS Ping: {ws} ms\n"
        f"Web Server: http://0.0.0.0:{WEB_PORT}",
        allowed_mentions=allowed_mentions
    )

def _read_free():
    try:
        out = subprocess.check_output(["free", "-m"], text=True).strip().split('\n')
        mem_line = out[1].split()
        total, used, free = mem_line[1], mem_line[2], mem_line[3]
        return f"{used}/{total}MB (free {free}MB)"
    except:
        return "n/a"

def _loadavg():
    try:
        la = os.getloadavg()
        return f"{la[0]:.2f} {la[1]:.2f} {la[2]:.2f}"
    except:
        return "n/a"

@bot.command(name="status")
async def status_cmd(ctx):
    ph = STATUS.get("phase", "")
    mdl = STATUS.get("model", "")
    pct = STATUS.get("pull_pct", 0)
    lyr = STATUS.get("pull_layer", "")
    done, tot = STATUS.get("pull_bytes", (0, 0))
    avg = STATUS.get("avg_sec", 0.0)
    err = STATUS.get("last_err", "")

    def _mb(n):
        try:
            return f"{n/1048576:.1f}MB"
        except:
            return "0MB"

    if ph == "pulling":
        msg = f"Phase: pulling\nModel: {mdl}\nTiến độ: {pct}% | Layer: {lyr} ({_mb(done)}/{_mb(tot)})"
    elif ph == "warmup":
        msg = f"Phase: warmup\nModel: {mdl}"
    elif ph == "answering":
        eta = max(8, int(avg)) if avg > 0 else 15
        msg = f"Phase: answering… (ETA ~{eta}s)"
    else:
        msg = f"Phase: {ph or 'ready'}\nModel: {mdl}\nAvg time: ~{avg:.1f}s"
    if err:
        msg += f"\nLast error: {err[:160]}"
    await ctx.reply(msg, allowed_mentions=allowed_mentions)

@bot.command(name="stats")
async def stats_cmd(ctx):
    eta = max(8, int(STATUS.get("avg_sec", 15))) if STATUS.get("avg_sec", 0) > 0 else 15
    slots = f"{len(ACTIVE_USERS)}/{MAX_ACTIVE_USERS}"
    msg = (
        f"Model: {STATUS.get('model', MODEL)}\nPhase: {STATUS.get('phase','ready')}\n"
        f"Avg time: ~{STATUS.get('avg_sec',0.0):.1f}s | ETA lúc bận: ~{eta}s\n"
        f"Slots: {slots}\nRAM: {_read_free()} | Load: {_loadavg()}\n"
        f"Profile: {PREFERRED_PROFILE}\nWeb: http://0.0.0.0:{WEB_PORT}"
    )
    await ctx.reply(msg, allowed_mentions=allowed_mentions)

@bot.command(name="pull")
async def pull_cmd(ctx, *, model: str):
    await ensure_session()
    STATUS["phase"] = "pulling"
    STATUS["model"] = model
    msg = await ctx.reply(f"{ctx.author.mention} Pull {model}: 0%", allowed_mentions=allowed_mentions)
    layers = {}
    last_pct = -1
    last_edit = 0
    try:
        async with session.post(
            f"http://127.0.0.1:{OLLAMA_PORT}/api/pull",
            json={"name": model, "stream": True}
        ) as resp:
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except:
                    continue
                if "error" in data:
                    STATUS["last_err"] = data["error"]
                    raise RuntimeError(data["error"])
                status = data.get("status", "")
                comp = int(data.get("completed") or 0)
                tot = int(data.get("total") or 0)
                dig = (data.get("digest") or "layer").replace("sha256:", "")[:12]
                if tot > 0:
                    layers[dig] = (comp, tot)
                total = sum(t for _, t in layers.values()) or tot
                done = sum(c for c, _ in layers.values()) or comp
                pct = int(done * 100 / total) if total else 0
                STATUS["pull_pct"] = pct
                STATUS["pull_layer"] = dig
                STATUS["pull_bytes"] = (done, total)
                now = time.time()
                if pct != last_pct or now - last_edit > 1.0:
                    last_pct = pct
                    last_edit = now
                    await msg.edit(
                        content=f"{ctx.author.mention} Pull {model}: {pct}% - {dig} ({done/1048576:.1f}MB/{tot/1048576:.1f}MB)"
                    )
        STATUS["phase"] = "ready"
        await msg.edit(content=f"{ctx.author.mention} Pull {model}: 100% ✅")
    except Exception as e:
        STATUS["last_err"] = str(e)
        await msg.edit(content=f"{ctx.author.mention} Pull lỗi: {e}")

@bot.command(name="warm")
async def warm_cmd(ctx):
    try:
        STATUS["phase"] = "warmup"
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": WARMUP_PROMPT}],
            "stream": False,
            "keep_alive": 300,
            "options": {"num_thread": (os.cpu_count() or 2), "num_ctx": 1536, "num_predict": 64}
        }
        r = requests.post(f"http://127.0.0.1:{OLLAMA_PORT}/api/chat", json=body, timeout=180)
        STATUS["phase"] = "ready"
        await ctx.reply(f"{ctx.author.mention} Warm status: {r.status_code}", allowed_mentions=allowed_mentions)
    except Exception as e:
        STATUS["last_err"] = str(e)
        await ctx.reply(f"{ctx.author.mention} Warm lỗi: {e}", allowed_mentions=allowed_mentions)

@bot.command(name="health")
async def health_cmd(ctx):
    try:
        r = requests.get(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", timeout=5)
        await ctx.reply(f"{ctx.author.mention} /api/tags: {r.status_code}", allowed_mentions=allowed_mentions)
    except Exception as e:
        await ctx.reply(f"{ctx.author.mention} Health lỗi: {e}", allowed_mentions=allowed_mentions)

@bot.command(name="log")
async def log_tail_cmd(ctx, n: int = 120):
    n = max(20, min(400, n))
    if not OLLAMA_LOG.exists():
        await ctx.reply(f"{ctx.author.mention} Chưa có log.", allowed_mentions=allowed_mentions)
        return
    tail = "".join(OLLAMA_LOG.read_text(errors="ignore").splitlines(True)[-n:])
    await ctx.reply(f"```{tail[-(DISCORD_MAX-10):]}```", allowed_mentions=allowed_mentions)

@bot.command(name="saveall")
async def saveall_cmd(ctx):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(DATA_DIR):
            for fn in files:
                p = Path(root) / fn
                arc = str(p.relative_to(DATA_DIR.parent))
                z.write(p, arcname=arc)
        if OLLAMA_LOG.exists():
            z.write(OLLAMA_LOG, arcname="logs/ollama.log")
    buf.seek(0)
    try:
        await ctx.reply(file=discord.File(buf, filename="bot_backup.zip"))
    except Exception as e:
        await ctx.reply(f"{ctx.author.mention} Không gửi được file: {e}", allowed_mentions=allowed_mentions)

@bot.command(name="help")
async def help_cmd(ctx):
    text = (
        "**Lệnh Bot**\n"
        "- Chat: @mention bot | !ai <nội dung> | DM bot (có thể kèm ảnh)\n"
        "- !describe (kèm ảnh): mô tả ảnh nhanh\n"
        "- !train status | !train list [n] | !train clear\n"
        "- !auto on|off | !remember <text> | !brain show|clear\n"
        "- !chatai @user: bot-to-bot 1 vòng\n"
        "- !status, !stats, !info, !health\n"
        "- !model [id] | !pull <model> | !warm\n"
        "- !clear | !log [n] | !saveall\n\n"
        f"**Web Endpoints** (Port {WEB_PORT}):\n"
        "- GET / → health check JSON (cho UptimeRobot)\n"
        "- GET /health → chi tiết health\n"
        "- GET /stats → statistics\n"
        "- GET /ping → simple ping\n\n"
        f"Model: {MODEL} | Slots: {MAX_ACTIVE_USERS}"
    )
    try:
        await ctx.reply(
            embed=discord.Embed(title="Help", description=text, color=0x5865F2),
            allowed_mentions=allowed_mentions
        )
    except:
        await ctx.reply(text, allowed_mentions=allowed_mentions)

@bot.command(name="web")
async def web_cmd(ctx):
    """Hiển thị thông tin web server"""
    await ctx.reply(
        f"{ctx.author.mention}\n"
        f"Web Server đang chạy tại: http://0.0.0.0:{WEB_PORT}\n"
        f"Endpoints:\n"
        f"- / (health check)\n"
        f"- /health\n"
        f"- /stats\n"
        f"- /ping\n\n"
        f"Để dùng với UptimeRobot, thêm URL: http://YOUR_DOMAIN:{WEB_PORT}/",
        allowed_mentions=allowed_mentions
    )

# ======== Main: Chạy Flask + Discord bot ========
async def main():
    """Khởi động cả Flask và Discord bot"""
    # Start Flask trong thread riêng
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print(f"✓ Flask web server started on port {WEB_PORT}")
    
    # Lấy token từ environment variable
    TOKEN = os.environ.get("DISCORD_TOKEN")
    if not TOKEN:
        raise RuntimeError("DISCORD_TOKEN environment variable not set!")
    
    print("Đang khởi động Discord bot...")
    print(f"Web server: http://0.0.0.0:{WEB_PORT}")
    await bot.start(TOKEN)

# Entry point
if __name__ == "__main__":
    try:
        # Kiểm tra token từ env trước
        TOKEN = os.environ.get("DISCORD_TOKEN")
        if not TOKEN:
            print("ERROR: DISCORD_TOKEN environment variable not set!")
            print("Please set it in Render dashboard or use: export DISCORD_TOKEN='your_token'")
            sys.exit(1)
        
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✓ Bot đã dừng")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
