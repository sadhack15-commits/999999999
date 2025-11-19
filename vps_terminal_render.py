#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ULTRA-FAST PUBLIC PROXY SERVER - RENDER OPTIMIZED
Fixed: ERR_TUNNEL_CONNECTION_FAILED on Render HTTPS routing
"""

import os
import socket
import select
import threading
from socketserver import ThreadingMixIn
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import ssl
import base64
import time

# ==================== CONFIG ====================
CONFIG = {
    'PROXY_PORT': int(os.environ.get('PORT', 10000)),
    'PROXY_PASSWORD': os.environ.get('PROXY_PASSWORD', ''),
    'MAX_CONNECTIONS': 500,
    'TIMEOUT': 15,
    'BUFFER_SIZE': 65536,
    'SOCKET_BUFFER': 262144,
}

# Stats
class Stats:
    def __init__(self):
        self.total = 0
        self.active = 0
        self.bytes = 0
        self.start = time.time()
        self.lock = threading.Lock()
    
    def inc_request(self):
        with self.lock:
            self.total += 1
            self.active += 1
    
    def dec_active(self):
        with self.lock:
            self.active -= 1
    
    def add_bytes(self, n):
        with self.lock:
            self.bytes += n

STATS = Stats()

# ==================== THREADED SERVER ====================
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    
    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, CONFIG['SOCKET_BUFFER'])
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, CONFIG['SOCKET_BUFFER'])
        super().server_bind()

# ==================== RENDER-OPTIMIZED PROXY HANDLER ====================
class RenderProxyHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    timeout = CONFIG['TIMEOUT']
    
    def do_CONNECT(self):
        """
        CONNECT method - Optimized for Render
        Note: May not work through Render HTTPS routing
        """
        STATS.inc_request()
        
        if CONFIG['PROXY_PASSWORD'] and not self._check_auth():
            self.send_error(407, 'Proxy Authentication Required')
            STATS.dec_active()
            return
        
        try:
            host, port = self.path.split(':')
            port = int(port)
            
            # Create optimized socket
            target = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            target.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            target.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, CONFIG['SOCKET_BUFFER'])
            target.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, CONFIG['SOCKET_BUFFER'])
            target.settimeout(CONFIG['TIMEOUT'])
            target.connect((host, port))
            
            # Send 200 Connection Established
            self.send_response(200, 'Connection Established')
            self.send_header('Proxy-Agent', 'Render-Proxy/1.0')
            self.end_headers()
            
            # Tunnel with select()
            self._tunnel_select(self.connection, target)
            
        except socket.timeout:
            self.send_error(504, 'Gateway Timeout')
        except ConnectionRefusedError:
            self.send_error(502, 'Connection Refused')
        except Exception as e:
            self.send_error(502, f"Bad Gateway: {str(e)}")
        finally:
            STATS.dec_active()
    
    def do_GET(self):
        if self.path == '/' or self.path.startswith('/?'):
            self._serve_web()
            return
        if self.path == '/api/stats':
            self._serve_stats()
            return
        if self.path == '/health' or self.path == '/ping':
            self._serve_health()
            return
        self._proxy_request()
    
    def do_POST(self):
        self._proxy_request()
    
    def do_HEAD(self):
        self._proxy_request()
    
    def do_PUT(self):
        self._proxy_request()
    
    def do_DELETE(self):
        self._proxy_request()
    
    def do_OPTIONS(self):
        """Handle OPTIONS for CORS"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()
    
    def _proxy_request(self):
        """HTTP/HTTPS proxy - Works better on Render"""
        STATS.inc_request()
        
        if CONFIG['PROXY_PASSWORD'] and not self._check_auth():
            self.send_error(407, 'Proxy Authentication Required')
            STATS.dec_active()
            return
        
        try:
            url = urlparse(self.path)
            
            # Determine target
            if url.scheme == 'https':
                host = url.netloc or url.path.split('/')[0]
                port = 443
                use_ssl = True
            elif url.scheme == 'http':
                host = url.netloc or url.path.split('/')[0]
                port = 80
                use_ssl = False
            else:
                # Assume HTTP for relative paths
                host = self.headers.get('Host', '').split(':')[0]
                port = 80
                use_ssl = False
            
            if ':' in host:
                host, port = host.rsplit(':', 1)
                port = int(port)
            
            if not host:
                self.send_error(400, 'Bad Request: No host specified')
                STATS.dec_active()
                return
            
            # Create optimized socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, CONFIG['SOCKET_BUFFER'])
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, CONFIG['SOCKET_BUFFER'])
            sock.settimeout(CONFIG['TIMEOUT'])
            sock.connect((host, port))
            
            # SSL wrap if needed
            if use_ssl:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                sock = ctx.wrap_socket(sock, server_hostname=host)
            
            # Build request
            path = url.path or '/'
            if url.query:
                path += '?' + url.query
            
            request_line = f"{self.command} {path} HTTP/1.1\r\n"
            
            # Forward headers
            headers = []
            for k, v in self.headers.items():
                if k.lower() not in ['proxy-connection', 'proxy-authorization', 'connection']:
                    headers.append(f"{k}: {v}\r\n")
            
            headers.append("Connection: close\r\n\r\n")
            
            # Send request
            request_data = request_line.encode() + ''.join(headers).encode()
            sock.sendall(request_data)
            
            # Forward body if present
            if 'Content-Length' in self.headers:
                length = int(self.headers['Content-Length'])
                body = self.rfile.read(length)
                sock.sendall(body)
                STATS.add_bytes(len(body))
            
            # Stream response
            while True:
                chunk = sock.recv(CONFIG['BUFFER_SIZE'])
                if not chunk:
                    break
                self.wfile.write(chunk)
                STATS.add_bytes(len(chunk))
            
            sock.close()
            
        except socket.timeout:
            self.send_error(504, 'Gateway Timeout')
        except ConnectionRefusedError:
            self.send_error(502, 'Connection Refused')
        except Exception as e:
            self.send_error(502, f"Bad Gateway: {str(e)}")
        finally:
            STATS.dec_active()
    
    def _tunnel_select(self, client, target):
        """Bidirectional tunnel with select()"""
        try:
            client.setblocking(False)
            target.setblocking(False)
            
            sockets = [client, target]
            
            while True:
                readable, _, exceptional = select.select(sockets, [], sockets, CONFIG['TIMEOUT'])
                
                if exceptional:
                    break
                
                if not readable:
                    break
                
                for sock in readable:
                    try:
                        data = sock.recv(CONFIG['BUFFER_SIZE'])
                        if not data:
                            return
                        
                        if sock is client:
                            target.sendall(data)
                        else:
                            client.sendall(data)
                        
                        STATS.add_bytes(len(data))
                        
                    except:
                        return
        except:
            pass
        finally:
            try:
                client.close()
            except:
                pass
            try:
                target.close()
            except:
                pass
    
    def _check_auth(self):
        """Fast auth check"""
        auth = self.headers.get('Proxy-Authorization', '')
        if not auth.startswith('Basic '):
            return False
        try:
            creds = base64.b64decode(auth[6:]).decode()
            _, pwd = creds.split(':', 1)
            return pwd == CONFIG['PROXY_PASSWORD']
        except:
            return False
    
    def _serve_health(self):
        """Health check for UptimeRobot"""
        uptime = time.time() - STATS.start
        data = (
            f'{{"status":"ok","uptime":{int(uptime)},'
            f'"active":{STATS.active},"total":{STATS.total}}}'
        ).encode()
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(data))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(data)
    
    def _serve_stats(self):
        """Stats API"""
        uptime = time.time() - STATS.start
        hours = int(uptime // 3600)
        mins = int((uptime % 3600) // 60)
        secs = int(uptime % 60)
        
        data = (
            f'{{"total_requests":{STATS.total},'
            f'"active_connections":{STATS.active},'
            f'"bytes_transferred":{STATS.bytes},'
            f'"uptime_seconds":{int(uptime)},'
            f'"uptime_readable":"{hours}h {mins}m {secs}s",'
            f'"bandwidth_mb":{round(STATS.bytes/1048576, 2)}}}'
        ).encode()
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(data))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(data)
    
    def _serve_web(self):
        """Web interface"""
        render_url = os.environ.get('RENDER_EXTERNAL_URL', 'your-app.onrender.com')
        if render_url.startswith('https://'):
            render_url = render_url.replace('https://', '')
        
        html = f'''<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚡ Ultra-Fast Proxy 24/7</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 20px; color: white;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        h1 {{ font-size: 2.5rem; text-align: center; margin-bottom: 1rem; }}
        .card {{
            background: rgba(255,255,255,0.1); backdrop-filter: blur(20px);
            border-radius: 20px; padding: 1.5rem; margin-bottom: 1.5rem;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .warning {{
            background: rgba(239, 68, 68, 0.2); border-left: 4px solid #ef4444;
            padding: 1rem; border-radius: 5px; margin: 1rem 0;
        }}
        .success {{
            background: rgba(16, 185, 129, 0.2); border-left: 4px solid #10b981;
            padding: 1rem; border-radius: 5px; margin: 1rem 0;
        }}
        .note {{
            background: rgba(251, 191, 36, 0.2); border-left: 4px solid #fbbf24;
            padding: 1rem; border-radius: 5px; margin: 1rem 0;
        }}
        code {{ background: rgba(0,0,0,0.3); padding: 2px 8px; border-radius: 3px; }}
        .stats {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem; margin: 1rem 0;
        }}
        .stat {{
            background: rgba(255,255,255,0.08); padding: 1.2rem; border-radius: 15px;
            text-align: center;
        }}
        .stat-value {{ font-size: 1.8rem; font-weight: 800; color: #10b981; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ Ultra-Fast Proxy</h1>
        
        <div class="card">
            <h2>⚠️ Lưu Ý Quan Trọng - Render Limitations</h2>
            
            <div class="warning">
                <strong>❌ CONNECT Method không hoạt động trên Render HTTPS!</strong><br><br>
                
                Render free tier chỉ hỗ trợ HTTP routing qua HTTPS wrapper.<br>
                Điều này có nghĩa:
                <ul style="margin-left: 1.5rem; margin-top: 0.5rem;">
                    <li>✅ HTTP proxy: <strong>HOẠT ĐỘNG</strong></li>
                    <li>✅ HTTPS websites qua HTTP proxy: <strong>HOẠT ĐỘNG</strong></li>
                    <li>❌ HTTPS CONNECT tunneling: <strong>KHÔNG HOẠT ĐỘNG</strong></li>
                </ul>
            </div>
            
            <div class="success">
                <strong>✅ Giải pháp:</strong><br><br>
                
                <strong>Cách 1: Dùng HTTP Proxy (Khuyến nghị)</strong><br>
                Protocol: <code>HTTP</code><br>
                Server: <code>{render_url}</code><br>
                Port: <code>443</code><br><br>
                
                <strong>Cách 2: Deploy trên VPS khác</strong><br>
                • Heroku (cũng có hạn chế tương tự)<br>
                • Railway, Fly.io<br>
                • VPS thật (AWS, DigitalOcean, Vultr)<br><br>
                
                <strong>Cách 3: Dùng cho HTTP sites only</strong><br>
                Chỉ proxy các trang HTTP (không phải HTTPS)
            </div>
        </div>

        <div class="card">
            <h2>📊 Thống Kê</h2>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value" id="total">0</div>
                    <div>Requests</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="active">0</div>
                    <div>Active</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="bandwidth">0 MB</div>
                    <div>Bandwidth</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="uptime">0s</div>
                    <div>Uptime</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>🎯 Setup UptimeRobot</h2>
            <div class="note">
                URL: <code>https://{render_url}/health</code><br>
                Interval: <code>5 minutes</code>
            </div>
        </div>
    </div>

    <script>
        async function updateStats() {{
            try {{
                const res = await fetch('/api/stats');
                const data = await res.json();
                document.getElementById('total').textContent = data.total_requests;
                document.getElementById('active').textContent = data.active_connections;
                document.getElementById('bandwidth').textContent = data.bandwidth_mb + ' MB';
                document.getElementById('uptime').textContent = data.uptime_readable;
            }} catch(e) {{}}
        }}
        updateStats();
        setInterval(updateStats, 2000);
    </script>
</body>
</html>'''.encode()
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(html))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(html)
    
    def log_message(self, format, *args):
        """Minimal logging"""
        pass

# ==================== MAIN ====================
def main():
    print("\n" + "="*60)
    print("⚡ RENDER-OPTIMIZED PROXY SERVER")
    print("="*60)
    
    try:
        server = ThreadedHTTPServer(('0.0.0.0', CONFIG['PROXY_PORT']), RenderProxyHandler)
        
        render_url = os.environ.get('RENDER_EXTERNAL_URL', 'localhost')
        
        print(f"\n🚀 Server: 0.0.0.0:{CONFIG['PROXY_PORT']}")
        print(f"🌐 URL: {render_url}")
        print(f"💚 Health: {render_url}/health")
        print(f"\n⚠️  NOTE: CONNECT method may not work on Render HTTPS")
        print(f"✅ Use HTTP proxy protocol instead")
        print(f"\n{'='*60}\n")
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    main()
