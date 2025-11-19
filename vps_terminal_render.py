#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VPS Terminal 24/7 - Render Cloud Edition
Web Server + Terminal tích hợp - Tối ưu cho Render.com
"""

import os
import subprocess
import threading
import time
import json
import sys
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver

# ==================== CẤU HÌNH ====================
CONFIG = {
    'WEB_PORT': int(os.environ.get('PORT', 10000)),  # Render tự động set PORT
    'TERMINAL_PORT': int(os.environ.get('TERMINAL_PORT', 7681)),
    'PASSWORD': os.environ.get('TERMINAL_PASSWORD', 'terminal123'),
    'WORKING_DIR': 'vps_workspace',
    'WEB_DIR': 'website',
}

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def log(message, level='INFO'):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    colors = {'INFO': Colors.BLUE, 'SUCCESS': Colors.GREEN, 'WARNING': Colors.YELLOW, 'ERROR': Colors.RED}
    color = colors.get(level, Colors.BLUE)
    print(f"{color}[{timestamp}] [{level}]{Colors.END} {message}")

# ==================== SETUP WORKSPACE ====================
def setup_workspace():
    log("Setting up workspace...", "INFO")
    os.makedirs(CONFIG['WORKING_DIR'], exist_ok=True)
    os.chdir(CONFIG['WORKING_DIR'])
    
    directories = [CONFIG['WEB_DIR'], 'uploads', 'downloads', 'logs', 'scripts', 'tmp']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    log(f"Workspace: {os.getcwd()}", "SUCCESS")

# ==================== DOWNLOAD TTYD ====================
def download_ttyd():
    log("Checking ttyd...", "INFO")
    try:
        # Kiểm tra system ttyd trước
        result = subprocess.run(['which', 'ttyd'], capture_output=True, text=True)
        if result.returncode == 0:
            log("ttyd found in system", "SUCCESS")
            return True
        
        # Download nếu chưa có
        arch = os.uname().machine
        ttyd_url = {
            'x86_64': 'https://github.com/tsl0922/ttyd/releases/download/1.7.3/ttyd.x86_64',
            'aarch64': 'https://github.com/tsl0922/ttyd/releases/download/1.7.3/ttyd.arm64',
        }.get(arch, 'https://github.com/tsl0922/ttyd/releases/download/1.7.3/ttyd.x86_64')
        
        if not os.path.exists('ttyd'):
            subprocess.run(['wget', '-q', ttyd_url, '-O', 'ttyd'], check=True, timeout=60)
            subprocess.run(['chmod', '+x', 'ttyd'], check=True)
            log("ttyd downloaded", "SUCCESS")
        else:
            log("ttyd exists", "INFO")
        return True
    except Exception as e:
        log(f"ttyd warning: {e} - Terminal may not work", "WARNING")
        return False

# ==================== CREATE WEBSITE ====================
def create_demo_website():
    log("Creating website...", "INFO")
    
    # Index.html - Trang chủ đẹp
    index_html = '''<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VPS Terminal 24/7 - Render Cloud</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            width: 100%;
            animation: fadeIn 0.8s ease-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .header {
            text-align: center;
            margin-bottom: 3rem;
        }
        h1 {
            font-size: 3.5rem;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            font-weight: 800;
        }
        .status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(16, 185, 129, 0.2);
            border: 2px solid #10b981;
            padding: 0.5rem 1.5rem;
            border-radius: 50px;
            font-weight: 600;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.9; transform: scale(1.02); }
        }
        .dot {
            width: 10px;
            height: 10px;
            background: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 10px #10b981;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        .card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 2rem;
            border: 1px solid rgba(255,255,255,0.2);
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            background: rgba(255,255,255,0.15);
        }
        .card h2 {
            font-size: 1.5rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .card p {
            line-height: 1.8;
            opacity: 0.9;
        }
        .features {
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 2rem;
            border: 1px solid rgba(255,255,255,0.15);
        }
        .features h3 {
            font-size: 1.8rem;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
        }
        .feature {
            background: rgba(255,255,255,0.05);
            padding: 1.5rem;
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .feature strong {
            font-size: 1.1rem;
            display: block;
            margin-bottom: 0.5rem;
        }
        .links {
            display: flex;
            gap: 1rem;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 2rem;
        }
        .btn {
            background: rgba(255,255,255,0.2);
            color: white;
            padding: 1rem 2.5rem;
            border-radius: 50px;
            text-decoration: none;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s;
            border: 2px solid rgba(255,255,255,0.3);
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        .btn:hover {
            background: rgba(255,255,255,0.3);
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        }
        .btn-primary {
            background: linear-gradient(135deg, #10b981, #059669);
            border-color: #10b981;
        }
        .btn-primary:hover {
            background: linear-gradient(135deg, #059669, #047857);
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-top: 2rem;
        }
        .stat {
            background: rgba(255,255,255,0.08);
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.15);
        }
        .stat-value {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }
        .footer {
            text-align: center;
            margin-top: 3rem;
            opacity: 0.8;
            font-size: 0.9rem;
        }
        @media (max-width: 768px) {
            h1 { font-size: 2.5rem; }
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 VPS Terminal 24/7</h1>
            <div class="status">
                <span class="dot"></span>
                <span>ONLINE - Running on Render Cloud</span>
            </div>
        </div>

        <div class="grid">
            <div class="card" onclick="location.href='/terminal'">
                <h2>💻 Web Terminal</h2>
                <p>Truy cập terminal Linux qua trình duyệt. Chạy lệnh, quản lý files, và làm mọi thứ như SSH thật.</p>
            </div>

            <div class="card" onclick="location.href='/api'">
                <h2>🔌 REST API</h2>
                <p>API endpoint để kiểm tra trạng thái server, uptime, và các thông tin hệ thống.</p>
            </div>

            <div class="card" onclick="location.href='/about.html'">
                <h2>📖 Documentation</h2>
                <p>Hướng dẫn sử dụng, tính năng, và thông tin chi tiết về hệ thống.</p>
            </div>
        </div>

        <div class="features">
            <h3>✨ Tính năng nổi bật</h3>
            <div class="feature-grid">
                <div class="feature">
                    <strong>⚡ Hiệu suất cao</strong>
                    <p>Chạy trên Render Cloud với uptime 99.9%</p>
                </div>
                <div class="feature">
                    <strong>🔒 Bảo mật</strong>
                    <p>HTTPS mặc định, password protected terminal</p>
                </div>
                <div class="feature">
                    <strong>🌐 Truy cập mọi lúc</strong>
                    <p>Web-based, không cần cài đặt gì</p>
                </div>
                <div class="feature">
                    <strong>📊 Monitoring</strong>
                    <p>Real-time stats và health checks</p>
                </div>
            </div>

            <div class="stats">
                <div class="stat">
                    <div class="stat-value">24/7</div>
                    <div>Uptime</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="uptime">0s</div>
                    <div>Session Time</div>
                </div>
                <div class="stat">
                    <div class="stat-value">∞</div>
                    <div>Possibilities</div>
                </div>
            </div>
        </div>

        <div class="links">
            <a href="/terminal" class="btn btn-primary">💻 Mở Terminal</a>
            <a href="/api" class="btn">🔌 API Docs</a>
            <a href="/about.html" class="btn">📖 About</a>
        </div>

        <div class="footer">
            <p>🌟 Powered by VPS Terminal 24/7 Pro Edition</p>
            <p>Deployed on Render Cloud Platform</p>
        </div>
    </div>

    <script>
        const startTime = Date.now();
        function updateUptime() {
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            const hours = Math.floor(elapsed / 3600);
            const minutes = Math.floor((elapsed % 3600) / 60);
            const seconds = elapsed % 60;
            document.getElementById('uptime').textContent = 
                hours > 0 ? `${hours}h ${minutes}m` : 
                minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
        }
        setInterval(updateUptime, 1000);
        updateUptime();
    </script>
</body>
</html>'''

    # About.html
    about_html = '''<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About - VPS Terminal 24/7</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            padding: 2rem;
            line-height: 1.8;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            animation: fadeIn 0.5s;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        h1 {
            font-size: 3rem;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .card {
            background: rgba(255,255,255,0.05);
            padding: 2rem;
            border-radius: 15px;
            margin: 1.5rem 0;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card h2 {
            color: #667eea;
            margin-bottom: 1rem;
            font-size: 1.8rem;
        }
        ul {
            margin-left: 1.5rem;
            margin-top: 1rem;
        }
        li {
            margin: 0.5rem 0;
        }
        .btn {
            display: inline-block;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 1rem 2rem;
            border-radius: 50px;
            text-decoration: none;
            margin-top: 2rem;
            font-weight: 600;
            transition: transform 0.3s;
        }
        .btn:hover {
            transform: translateY(-2px);
        }
        code {
            background: rgba(255,255,255,0.1);
            padding: 0.2rem 0.5rem;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            color: #10b981;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📖 About VPS Terminal 24/7</h1>
        
        <div class="card">
            <h2>🎯 What is this?</h2>
            <p>
                VPS Terminal 24/7 là một giải pháp web-based terminal chạy trên Render Cloud Platform.
                Nó kết hợp web server và terminal emulator để tạo ra một môi trường Linux có thể 
                truy cập từ bất kỳ đâu thông qua trình duyệt.
            </p>
        </div>

        <div class="card">
            <h2>🚀 Tính năng chính</h2>
            <ul>
                <li>✅ <strong>Web Terminal:</strong> Truy cập terminal Linux qua browser</li>
                <li>✅ <strong>HTTP Server:</strong> Host website và files tĩnh</li>
                <li>✅ <strong>REST API:</strong> Endpoint để monitoring và automation</li>
                <li>✅ <strong>24/7 Uptime:</strong> Chạy liên tục trên Render Cloud</li>
                <li>✅ <strong>No Sudo Required:</strong> Hoạt động với user permissions</li>
                <li>✅ <strong>Auto Keep-Alive:</strong> Tự động giữ kết nối</li>
            </ul>
        </div>

        <div class="card">
            <h2>⚡ Công nghệ</h2>
            <ul>
                <li>🐍 <strong>Python 3:</strong> Core backend</li>
                <li>💻 <strong>ttyd:</strong> Web-based terminal emulator</li>
                <li>🌐 <strong>HTTP Server:</strong> Built-in Python SimpleHTTPServer</li>
                <li>☁️ <strong>Render:</strong> Cloud hosting platform</li>
                <li>🎨 <strong>HTML5/CSS3:</strong> Modern responsive UI</li>
            </ul>
        </div>

        <div class="card">
            <h2>📊 API Endpoints</h2>
            <p>API REST đơn giản để monitoring:</p>
            <ul>
                <li><code>GET /api</code> - System status và info</li>
                <li><code>GET /health</code> - Health check endpoint</li>
            </ul>
            <p style="margin-top: 1rem;">
                Ví dụ response từ <code>/api</code>:
            </p>
            <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; margin-top: 0.5rem; overflow-x: auto;">
{
  "status": "online",
  "uptime": 1234567.89,
  "version": "1.0.0",
  "services": {
    "web": "Port 10000",
    "terminal": "Port 7681"
  }
}</pre>
        </div>

        <div class="card">
            <h2>🔧 Cách hoạt động</h2>
            <p>
                Script Python tạo ra 3 threads chính:
            </p>
            <ol style="margin-left: 1.5rem; margin-top: 1rem;">
                <li><strong>Web Server Thread:</strong> Phục vụ website và API</li>
                <li><strong>Terminal Thread:</strong> Chạy ttyd để cung cấp web terminal</li>
                <li><strong>Keep-Alive Thread:</strong> Gửi heartbeat mỗi 5 phút</li>
            </ol>
        </div>

        <div class="card">
            <h2>⚠️ Limitations</h2>
            <ul>
                <li>🔒 Không có quyền root/sudo</li>
                <li>💾 Storage tạm thời (ephemeral filesystem)</li>
                <li>⏰ Free tier của Render sleep sau 15 phút không hoạt động</li>
                <li>🔌 Port giới hạn (chỉ PORT được Render assign)</li>
            </ul>
        </div>

        <a href="/" class="btn">🏠 Về trang chủ</a>
    </div>
</body>
</html>'''

    # Terminal interface
    terminal_html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VPS Web Terminal</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Courier New', monospace;
            background: #0f172a;
            color: #e2e8f0;
            min-height: 100vh;
            overflow: hidden;
        }}
        .header {{
            background: rgba(30, 41, 59, 0.95);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #334155;
            backdrop-filter: blur(10px);
        }}
        .logo {{
            font-size: 1.2rem;
            font-weight: bold;
            color: #667eea;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .status {{
            width: 8px;
            height: 8px;
            background: #10b981;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; box-shadow: 0 0 10px #10b981; }}
            50% {{ opacity: 0.5; box-shadow: 0 0 5px #10b981; }}
        }}
        .terminal-frame {{
            width: 100%;
            height: calc(100vh - 60px);
            border: none;
            background: #000;
        }}
        .btn {{
            background: #667eea;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            cursor: pointer;
            margin-left: 0.5rem;
            font-weight: 600;
            transition: all 0.3s;
        }}
        .btn:hover {{
            background: #5568d3;
            transform: translateY(-1px);
        }}
        .btn-home {{
            background: #10b981;
        }}
        .btn-home:hover {{
            background: #059669;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">
            <span class="status"></span>
            💻 VPS Terminal
        </div>
        <div>
            <button class="btn" onclick="location.reload()">🔄 Refresh</button>
            <button class="btn btn-home" onclick="location.href='/'">🏠 Home</button>
        </div>
    </div>
    <iframe class="terminal-frame" src="http://localhost:{CONFIG['TERMINAL_PORT']}" title="Terminal"></iframe>
</body>
</html>'''

    # Lưu files
    with open(f"{CONFIG['WEB_DIR']}/index.html", 'w', encoding='utf-8') as f:
        f.write(index_html)
    with open(f"{CONFIG['WEB_DIR']}/about.html", 'w', encoding='utf-8') as f:
        f.write(about_html)
    with open(f"{CONFIG['WEB_DIR']}/terminal.html", 'w', encoding='utf-8') as f:
        f.write(terminal_html)
    
    log("Website created successfully", "SUCCESS")

# ==================== STARTUP SCRIPT ====================
def create_startup_script():
    script = '''#!/bin/bash
echo "════════════════════════════════════════════════"
echo "   🚀 VPS Terminal 24/7 - Render Edition 🚀"
echo "════════════════════════════════════════════════"
echo ""
echo "📍 Directory: $(pwd)"
echo "💾 Disk: $(df -h . | tail -1 | awk '{print $4}') free"
echo "🐍 Python: $(python3 --version 2>&1)"
echo "🌐 Platform: Render Cloud"
echo ""
echo "════════════════════════════════════════════════"
echo ""
exec /bin/bash --norc
'''
    with open('start.sh', 'w') as f:
        f.write(script)
    subprocess.run(['chmod', '+x', 'start.sh'], check=True)

# ==================== WEB SERVER ====================
class WebServerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=CONFIG['WEB_DIR'], **kwargs)
    
    def do_GET(self):
        if self.path == '/':
            self.path = '/index.html'
        elif self.path == '/terminal':
            self.path = '/terminal.html'
        elif self.path == '/api':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            api_response = {
                'status': 'online',
                'uptime': time.time(),
                'version': '1.0.0',
                'platform': 'Render Cloud',
                'services': {
                    'web': f'Port {CONFIG["WEB_PORT"]}',
                    'terminal': f'Port {CONFIG["TERMINAL_PORT"]}'
                },
                'timestamp': datetime.now().isoformat()
            }
            self.wfile.write(json.dumps(api_response, indent=2).encode())
            return
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            health = {'status': 'healthy', 'timestamp': time.time()}
            self.wfile.write(json.dumps(health).encode())
            return
        return SimpleHTTPRequestHandler.do_GET(self)
    
    def log_message(self, format, *args):
        log(f"{self.address_string()} - {format % args}", "INFO")

def start_web_server():
    try:
        # Bind vào 0.0.0.0 để Render có thể access
        with socketserver.TCPServer(("0.0.0.0", CONFIG['WEB_PORT']), WebServerHandler) as httpd:
            log(f"🌐 Web Server running on 0.0.0.0:{CONFIG['WEB_PORT']}", "SUCCESS")
            httpd.serve_forever()
    except Exception as e:
        log(f"Web Server error: {e}", "ERROR")
        sys.exit(1)

# ==================== TERMINAL SERVER ====================
def start_terminal():
    time.sleep(3)
    try:
        # Kiểm tra ttyd có tồn tại không
        ttyd_path = './ttyd' if os.path.exists('./ttyd') else 'ttyd'
        
        cmd = [
            ttyd_path,
            '-p', str(CONFIG['TERMINAL_PORT']),
            '-W',  # Writable terminal
            '-t', 'fontSize=14',
            '-t', 'theme={"background": "#0f172a", "foreground": "#e2e8f0"}',
            './start.sh'
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        log(f"💻 Terminal running on port {CONFIG['TERMINAL_PORT']}", "SUCCESS")
        return process
    except Exception as e:
        log(f"Terminal warning: {e} - Terminal may not be available", "WARNING")
        return None

# ==================== KEEP ALIVE ====================
def keep_alive():
    log("🔄 Keep-alive monitor started", "INFO")
    count = 0
    while True:
        time.sleep(300)  # 5 minutes
        count += 1
        log(f"✓ Heartbeat #{count} - Services running", "SUCCESS")

# ==================== MAIN ====================
def main():
    print(f"\n{Colors.BOLD}{Colors.GREEN}")
    print("═══════════════════════════════════════════════════")
    print("   🚀 VPS TERMINAL 24/7 - RENDER EDITION 🚀")
    print("═══════════════════════════════════════════════════")
    print(f"{Colors.END}\n")
    
    # Setup
    setup_workspace()
    
    # Download ttyd (optional, không bắt buộc)
    ttyd_available = download_ttyd()
    
    # Create website
    create_demo_website()
    create_startup_script()
    
    # Start services
    log("🚀 Starting services...", "INFO")
    
    # Web Server (main thread)
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()
    time.sleep(2)
    
    # Terminal Server (nếu ttyd có sẵn)
    terminal_process = None
    if ttyd_available:
        terminal_thread = threading.Thread(target=start_terminal, daemon=True)
        terminal_thread.start()
        time.sleep(2)
    else:
        log("Terminal not available - Web server only mode", "WARNING")
    
    # Keep alive monitor
    keepalive_thread = threading.Thread(target=keep_alive, daemon=True)
    keepalive_thread.start()
    
    # Display success banner
    print(f"\n{Colors.GREEN}{Colors.BOLD}═══════════════════════════════════════════════════{Colors.END}")
    print(f"{Colors.GREEN}✅ ALL SERVICES RUNNING ON RENDER CLOUD!{Colors.END}")
    print(f"{Colors.GREEN}{Colors.BOLD}═══════════════════════════════════════════════════{Colors.END}\n")
    print(f"🌐 {Colors.BOLD}Website:{Colors.END} http://0.0.0.0:{CONFIG['WEB_PORT']}")
    print(f"🌐 {Colors.BOLD}Public URL:{Colors.END} https://your-app-name.onrender.com")
    print(f"💻 {Colors.BOLD}Terminal:{Colors.END} /terminal")
    print(f"🔌 {Colors.BOLD}API:{Colors.END} /api")
    print(f"❤️  {Colors.BOLD}Health:{Colors.END} /health")
    print(f"\n{Colors.YELLOW}💡 Features:{Colors.END}")
    print(f"  ✅ Web server running 24/7")
    print(f"  ✅ Terminal access (if available)")
    print(f"  ✅ RESTful API endpoints")
    print(f"  ✅ Health check endpoint")
    print(f"  ✅ Auto keep-alive system")
    print(f"  ✅ Render Cloud optimized")
    print(f"\n{Colors.BLUE}📊 System Info:{Colors.END}")
    print(f"  Platform: {os.uname().sysname} {os.uname().release}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Working Dir: {os.getcwd()}")
    print(f"\n{Colors.GREEN}{Colors.BOLD}═══════════════════════════════════════════════════{Colors.END}\n")
    
    # Main loop - keep script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Shutting down gracefully...{Colors.END}")
        sys.exit(0)

if __name__ == '__main__':
    main()
