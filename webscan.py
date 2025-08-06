# linecam.py
# Webカメラ用 簡易ラインスキャナ（カラー対応）．
# 追加：カラー出力トグル．RGBキャンバスでの積層，PNG/PDFカラー保存．

import base64
import io
from datetime import datetime

import cv2
import img2pdf
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, Response, send_file

app = Flask(__name__)

# 追加．レスポンス共通ヘッダでキャッシュ抑止
@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp

# ラインスキャン状態
RUNNING = False
MODE = "vertical"   # "vertical" or "horizontal"
LINE_POS = 0.5      # 0.0..1.0 正規化位置
STRIPE = 3          # 太さpx
DOWNSCALE = 1       # 出力縮小倍率
ROTATE = 0          # 0,90,180,270
COLOR = True        # TrueでRGB積層，Falseでグレースケール

# 積層キャンバス
# COLOR=True なら shape(H, W, 3)，False なら shape(H, W)
CANVAS = None
LAST_FRAME_SHAPE = None  # (h, w)

def decode_frame(data_url: str):
    _, b64 = data_url.split(",", 1)
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR

def ensure_canvas(h, w):
    """フレームサイズに応じて空キャンバスを初期化する．"""
    global CANVAS, LAST_FRAME_SHAPE
    LAST_FRAME_SHAPE = (h, w)
    if CANVAS is not None:
        return
    if MODE == "vertical":
        # 最初は幅0，高さは縮小後
        H = max(1, h // DOWNSCALE)
        if COLOR:
            CANVAS = np.zeros((H, 0, 3), dtype=np.uint8)
        else:
            CANVAS = np.zeros((H, 0), dtype=np.uint8)
    else:
        # 最初は高さ0，幅は縮小後
        W = max(1, w // DOWNSCALE)
        if COLOR:
            CANVAS = np.zeros((0, W, 3), dtype=np.uint8)
        else:
            CANVAS = np.zeros((0, W), dtype=np.uint8)

def extract_stripe_rgb(rgb, line_pos, stripe, mode):
    """RGB画像から固定ラインのストライプを抽出して平均し，1ピクセル幅（縦or横）のRGBストリップにする．"""
    h, w, _ = rgb.shape
    s = max(1, int(stripe))
    if mode == "vertical":
        x = int(np.clip(line_pos * w, 0, w - 1))
        half = s // 2
        x0 = max(0, x - half)
        x1 = min(w, x + half + 1)
        # (h, s, 3) → 列方向平均 → (h, 3) → (h, 1, 3)
        col = rgb[:, x0:x1, :].mean(axis=1).astype(np.uint8)
        return col.reshape(h, 1, 3)
    else:
        y = int(np.clip(line_pos * h, 0, h - 1))
        half = s // 2
        y0 = max(0, y - half)
        y1 = min(h, y + half + 1)
        # (s, w, 3) → 行方向平均 → (w, 3) → (1, w, 3)
        row = rgb[y0:y1, :, :].mean(axis=0).astype(np.uint8)
        return row.reshape(1, w, 3)

def extract_stripe_gray(gray, line_pos, stripe, mode):
    """グレースケール版のストリップ抽出．"""
    h, w = gray.shape
    s = max(1, int(stripe))
    if mode == "vertical":
        x = int(np.clip(line_pos * w, 0, w - 1))
        half = s // 2
        x0 = max(0, x - half)
        x1 = min(w, x + half + 1)
        col = gray[:, x0:x1].mean(axis=1).astype(np.uint8)  # (h,)
        return col.reshape(h, 1)
    else:
        y = int(np.clip(line_pos * h, 0, h - 1))
        half = s // 2
        y0 = max(0, y - half)
        y1 = min(h, y + half + 1)
        row = gray[y0:y1, :].mean(axis=0).astype(np.uint8)  # (w,)
        return row.reshape(1, w)

def downscale_strip(strip, k):
    """ストリップを縮小する．縦モードは高さ方向，横モードは幅方向を縮小．"""
    if k <= 1:
        return strip
    if MODE == "vertical":
        # 形状 (H, 1[, 3]) → 高さH//k
        new_h = max(1, strip.shape[0] // k)
        return cv2.resize(strip, (1, new_h), interpolation=cv2.INTER_AREA)
    else:
        # 形状 (1, W[, 3]) → 幅W//k
        new_w = max(1, strip.shape[1] // k)
        return cv2.resize(strip, (new_w, 1), interpolation=cv2.INTER_AREA)

def append_strip(strip):
    """キャンバスにストリップを追加連結する．"""
    global CANVAS
    if MODE == "vertical":
        # 高さが合わない場合は作り直し
        if strip.shape[0] != CANVAS.shape[0]:
            if strip.ndim == 3:
                CANVAS = np.zeros((strip.shape[0], 0, 3), dtype=np.uint8)
            else:
                CANVAS = np.zeros((strip.shape[0], 0), dtype=np.uint8)
        CANVAS = np.hstack([CANVAS, strip])
    else:
        if strip.ndim == 3:
            if CANVAS.ndim != 3 or strip.shape[1] != CANVAS.shape[1]:
                CANVAS = np.zeros((0, strip.shape[1], 3), dtype=np.uint8)
        else:
            if strip.shape[1] != CANVAS.shape[1]:
                CANVAS = np.zeros((0, strip.shape[1]), dtype=np.uint8)
        CANVAS = np.vstack([CANVAS, strip])

@app.route("/")
def index():
    html = """
<!doctype html>
<meta charset="utf-8">
<title>LineCam Emulator (Color)</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:24px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}
.stage{position:relative;max-width:640px}
video,canvas.overlay{width:100%;background:#000;display:block}
canvas#hidden{display:none}
.btns{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}
label{display:inline-flex;gap:6px;align-items:center;margin-right:10px}
small{color:#555}
#preview{border:1px solid #ddd;width:100%;min-height:120px;display:block}
.badge{position:absolute;left:8px;top:8px;background:rgba(0,0,0,.6);color:#fff;padding:2px 6px;border-radius:6px;font-size:12px}
</style>

<h1>簡易ラインスキャナ エミュレータ（カラー対応）</h1>
<p>固定ラインを通過する画素を時間方向に積層します．動画上にライン位置のオーバーレイを表示します．</p>

<div class="btns">
  <label>モード
    <select id="mode">
      <option value="vertical">縦ラインで横に積層</option>
      <option value="horizontal">横ラインで縦に積層</option>
    </select>
  </label>
  <label>ライン位置<input id="pos" type="range" min="0" max="1" step="0.001" value="0.5"></label>
  <label>太さpx<input id="stripe" type="number" min="1" max="50" value="3" style="width:80px"></label>
  <label>縮小<input id="down" type="number" min="1" max="8" value="1" style="width:60px"></label>
  <label>回転
    <select id="rot"><option>0</option><option>90</option><option>180</option><option>270</option></select>
  </label>
  <label><input id="color" type="checkbox" checked> カラー出力</label>
  <label>送信FPS<input id="fps" type="number" min="1" max="60" value="20" style="width:70px"></label>
</div>

<div class="btns">
  <button id="start">Start</button>
  <button id="stop" disabled>Stop</button>
  <button id="reset">Reset</button>
  <button id="savepng">PNG保存</button>
  <button id="savepdf">PDF保存</button>
</div>
<small>ヒント．露光を短くし，拡散光で均一照明にすると軌跡が鮮明になります．</small>

<div class="grid">
  <div class="stage">
    <div class="badge" id="badge">x=50．width=3</div>
    <video id="video" autoplay playsinline></video>
    <canvas id="overlay" class="overlay"></canvas>
    <canvas id="hidden"></canvas>
  </div>
  <div>
    <img id="preview" src="" alt="accumulated image preview">
  </div>
</div>

<script>
const v=document.getElementById('video');
const overlay=document.getElementById('overlay');
const hidden=document.getElementById('hidden');
const badge=document.getElementById('badge');
const mode=document.getElementById('mode');
const pos=document.getElementById('pos');
const stripe=document.getElementById('stripe');
const down=document.getElementById('down');
const rot=document.getElementById('rot');
const color=document.getElementById('color');
const fps=document.getElementById('fps');

const startBtn=document.getElementById('start');
const stopBtn=document.getElementById('stop');
const resetBtn=document.getElementById('reset');
const pngBtn=document.getElementById('savepng');
const pdfBtn=document.getElementById('savepdf');
const preview=document.getElementById('preview');

let sendTimer=null, previewTimer=null, overlayRAF=null;

async function setup(){
  const stream = await navigator.mediaDevices.getUserMedia({
    video:{frameRate:{ideal:60},width:{ideal:1280},height:{ideal:720}}, audio:false
  });
  v.srcObject = stream;
  await new Promise(r => v.onloadedmetadata = r);
  hidden.width = v.videoWidth; hidden.height = v.videoHeight;
  overlay.width = v.videoWidth; overlay.height = v.videoHeight;
  drawOverlay(); loopOverlay();
}
setup().catch(e => alert('カメラ初期化に失敗しました．権限を確認してください．'));

function frameDataURL(){
  const ctx = hidden.getContext('2d');
  ctx.drawImage(v, 0, 0, hidden.width, hidden.height);
  return hidden.toDataURL('image/jpeg', 0.7);
}

async function postJSON(url, body){
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  return r.ok ? r.json() : {error:true};
}

function drawOverlay(){
  const ctx = overlay.getContext('2d');
  const w = overlay.width, h = overlay.height;
  ctx.clearRect(0,0,w,h);
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(0,180,255,0.9)';
  ctx.fillStyle = 'rgba(0,180,255,0.2)';
  const s = Math.max(1, parseInt(stripe.value,10));
  const p = Math.min(1, Math.max(0, parseFloat(pos.value)));
  if(mode.value === 'vertical'){
    const x = Math.round(p * w);
    const half = Math.floor(s/2);
    const x0 = Math.max(0, x - half), x1 = Math.min(w, x + half + 1);
    ctx.fillRect(x0, 0, Math.max(1,x1-x0), h);
    ctx.beginPath(); ctx.moveTo(x+0.5, 0); ctx.lineTo(x+0.5, h); ctx.stroke();
    badge.textContent = `x=${x}．width=${Math.max(1,x1-x0)}`;
  }else{
    const y = Math.round(p * h);
    const half = Math.floor(s/2);
    const y0 = Math.max(0, y - half), y1 = Math.min(h, y + half + 1);
    ctx.fillRect(0, y0, w, Math.max(1,y1-y0));
    ctx.beginPath(); ctx.moveTo(0, y+0.5); ctx.lineTo(w, y+0.5); ctx.stroke();
    badge.textContent = `y=${y}．height=${Math.max(1,y1-y0)}`;
  }
  ctx.restore();
}
function loopOverlay(){ drawOverlay(); overlayRAF = requestAnimationFrame(loopOverlay); }
[mode, pos, stripe].forEach(el => el.addEventListener('input', drawOverlay));

async function startSending(){
  const cfg = {
    mode: mode.value,
    pos: parseFloat(pos.value),
    stripe: parseInt(stripe.value,10),
    down: parseInt(down.value,10),
    rotate: parseInt(rot.value,10),
    color: color.checked ? 1 : 0
  };
  // ここを await に変更
  const ok = await postJSON('/start', cfg);
  if(!ok || ok.error){ alert('開始に失敗しました．'); return; }

  const period = Math.max(5, Math.round(1000 / Math.min(60, Math.max(1, parseInt(fps.value,10)))));
  if(sendTimer) clearInterval(sendTimer);
  sendTimer = setInterval(async () => {
    const durl = frameDataURL();
    // 送信失敗を握りつぶさず軽くリトライ
    try { await postJSON('/stream', {frame: durl}); } catch(e) {}
  }, period);

  if(previewTimer) clearInterval(previewTimer);
  // プレビューは 200ms → 400ms にして負荷を半減
  previewTimer = setInterval(() => { preview.src = '/image?ts=' + Date.now(); }, 400);
}

startBtn.onclick = () => {
  startBtn.disabled = true; stopBtn.disabled = false;
  startSending();
};
stopBtn.onclick = async () => {
  await fetch('/stop', {method:'POST'});
  if(sendTimer) clearInterval(sendTimer);
  if(previewTimer) clearInterval(previewTimer);
  sendTimer=null; previewTimer=null;
  startBtn.disabled = false; stopBtn.disabled = true;
  preview.src = '/image?ts=' + Date.now();
};

function restartIfRunning(){
  if(!stopBtn.disabled){ // 稼働中
    startSending();      // タイマーを組み直す
  }
}
[mode, rot, down, color].forEach(el => el.addEventListener('change', restartIfRunning));
fps.addEventListener('change', restartIfRunning);

resetBtn.onclick = async () => { await fetch('/reset', {method:'POST'}); preview.src = ''; };
pngBtn.onclick = async () => {
  const r = await fetch('/save_png', {method:'POST'});
  if(r.ok){
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'linecam_'+new Date().toISOString().replace(/[:.]/g,'-')+'.png';
    document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  }
};
pdfBtn.onclick = async () => {
  const r = await fetch('/save_pdf', {method:'POST'});
  if(r.ok){
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'linecam_'+new Date().toISOString().replace(/[:.]/g,'-')+'.pdf';
    document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  }
};
</script>
"""
    return Response(html, mimetype="text/html")

@app.route("/start", methods=["POST"])
def start():
    global MODE, LINE_POS, STRIPE, DOWNSCALE, ROTATE, COLOR, CANVAS, CANVAS_WIDTH, PIXELS_COUNT, STRIPS
    req = request.get_json() or {}
    MODE = req.get("mode", "vertical")
    LINE_POS = max(0, min(1, req.get("pos", 0.5)))
    STRIPE = max(1, min(50, req.get("stripe", 3)))
    DOWNSCALE = max(1, min(8, req.get("down", 1)))
    ROTATE = req.get("rotate", 0) if req.get("rotate") in (0, 90, 180, 270) else 0
    COLOR = bool(req.get("color", 1))  # カラーフラグを設定（デフォルト：カラーON）
    
    # キャンバスを初期化
    CANVAS = None
    CANVAS_WIDTH = 0
    PIXELS_COUNT = 0
    STRIPS = []
    return jsonify({"ok": 1, "mode": MODE, "pos": LINE_POS, "stripe": STRIPE, "downscale": DOWNSCALE, "rotate": ROTATE, "color": COLOR})

@app.route("/stop", methods=["POST"])
def stop():
    global RUNNING
    RUNNING = False
    return jsonify({"ok": 1})

@app.route("/reset", methods=["POST"])
def reset():
    global CANVAS
    CANVAS = None
    return jsonify({"ok": 1})

@app.route("/stream", methods=["POST"])
def stream():
    global RUNNING, CANVAS
    d = request.get_json()
    # 受信が来たら防御的に稼働状態へ
    if not RUNNING:
        RUNNING = True
        CANVAS = None  # 設定ずれを避けるため毎回初期化
    
    bgr = decode_frame(d["frame"])
    if ROTATE in (90, 180, 270):
        bgr = cv2.rotate(bgr, {90:cv2.ROTATE_90_CLOCKWISE, 180:cv2.ROTATE_180, 270:cv2.ROTATE_90_COUNTERCLOCKWISE}[ROTATE])
    
    h, w = bgr.shape[:2]
    ensure_canvas(h, w)
    
    # カラーモードに応じてストライプ抽出
    if COLOR:
        # RGB モード：色情報を保持
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        strip = extract_stripe_rgb(rgb, LINE_POS, STRIPE, MODE)
    else:
        # グレースケールモード：従来通り
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        strip = extract_stripe_gray(gray, LINE_POS, STRIPE, MODE)
    
    # ストライプ縮小処理
    if DOWNSCALE > 1:
        strip = downscale_strip(strip, DOWNSCALE)
    
    append_strip(strip)
    return jsonify({"ok": 1, "running": 1, "pixels": int(strip.size)})

@app.route("/image")
def image():
    if CANVAS is None or CANVAS.size == 0:
        blank = Image.new("L", (640, 120), color=255)
        buf = io.BytesIO(); blank.save(buf, format="PNG"); buf.seek(0)
        return send_file(buf, mimetype="image/png")
    img = CANVAS.copy()
    # 表示用に過度な巨大化を抑制
    MAXW, MAXH = 4000, 2000
    if img.ndim == 3:
        h, w, _ = img.shape
    else:
        h, w = img.shape
    scale = min(MAXW / max(1, w), MAXH / max(1, h), 1.0)
    if scale < 1.0:
        if img.ndim == 3:
            img = cv2.resize(img, (max(1,int(w*scale)), max(1,int(h*scale))), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (max(1,int(w*scale)), max(1,int(h*scale))), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(img)
    buf = io.BytesIO(); pil.save(buf, format="PNG"); buf.seek(0)
    return send_file(buf, mimetype="image/png")

@app.route("/save_png", methods=["POST"])
def save_png():
    if CANVAS is None or CANVAS.size == 0:
        return jsonify({"error": "empty"}), 400
    pil = Image.fromarray(CANVAS)
    buf = io.BytesIO(); pil.save(buf, format="PNG"); buf.seek(0)
    name = f"linecam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    return send_file(buf, mimetype="image/png", as_attachment=True, download_name=name)

@app.route("/save_pdf", methods=["POST"])
def save_pdf():
    if CANVAS is None or CANVAS.size == 0:
        return jsonify({"error": "empty"}), 400
    # カラー画像の場合はRGBからLに変換、グレースケールはそのまま
    if CANVAS.ndim == 3:
        pil = Image.fromarray(CANVAS).convert("L")
    else:
        pil = Image.fromarray(CANVAS, mode="L")
    jpg = io.BytesIO(); pil.save(jpg, format="JPEG", quality=95); jpg.seek(0)
    layout = img2pdf.get_layout_fun(
        img2pdf.mm_to_pt(210), img2pdf.mm_to_pt(297),
        img2pdf.mm_to_pt(210), img2pdf.mm_to_pt(297),
        img2pdf.Alignment.CENTER
    )
    pdf = img2pdf.convert([jpg.read()], layout_fun=layout)
    buf = io.BytesIO(pdf)
    name = f"linecam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name=name)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
