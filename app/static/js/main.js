/**
 * FaceSentrix - Advanced HUD Controller
 */

const video = document.getElementById('webcam-feed');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const metricContainer = document.getElementById('metric-container');
const toggleBtn = document.getElementById('toggle-cam');
const captureBtn = document.getElementById('capture-btn');

const emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"];
let isStreaming = false;
let stream = null;

// 1. Initialize Video Stream
async function startWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: { ideal: 1280 }, height: { ideal: 720 } } 
        });
        video.srcObject = stream;
        isStreaming = true;
        
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            document.getElementById('res-val').innerText = `${video.videoWidth}X${video.videoHeight}`;
            processFrame();
        };
    } catch (err) {
        console.error("Critical System Error:", err);
        alert("HUD Initialization Failed: Webcam access denied.");
    }
}

// 2. High-Performance Processing Loop
async function processFrame() {
    if (!isStreaming) return;

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0);

    const base64Image = tempCanvas.toDataURL('image/jpeg', 0.6);
    const startTime = performance.now();

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Image })
        });

        const data = await response.json();
        const endTime = performance.now();
        
        if (data.success) {
            renderHUD(data.results);
            updateBiometrics(data.results, (endTime - startTime).toFixed(0));
        }
    } catch (err) {
        console.error("Data Stream Interrupted:", err);
    }

    setTimeout(processFrame, 60); // Target ~15 FPS for smooth HUD feel
}

// 3. Cyberpunk HUD Rendering
function renderHUD(results) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('face-count').innerText = String(results.length).padStart(2, '0');

    results.forEach(res => {
        const [x, y, w, h] = res.box;
        const color = getEmotionColor(res.emotion);
        
        // Target Brackets (Corners only)
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        const len = w * 0.2;
        
        // TL
        ctx.beginPath(); ctx.moveTo(x, y + len); ctx.lineTo(x, y); ctx.lineTo(x + len, y); ctx.stroke();
        // TR
        ctx.beginPath(); ctx.moveTo(x + w - len, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + len); ctx.stroke();
        // BL
        ctx.beginPath(); ctx.moveTo(x, y + h - len); ctx.lineTo(x, y + h); ctx.lineTo(x + len, y + h); ctx.stroke();
        // BR
        ctx.beginPath(); ctx.moveTo(x + w - len, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - len); ctx.stroke();

        // HUD Data Label
        ctx.fillStyle = color;
        ctx.font = '700 12px "JetBrains Mono"';
        const labelText = `ID:TARGET_ALPHA // ${res.emotion.toUpperCase()} [${(res.confidence * 100).toFixed(0)}%]`;
        ctx.fillText(labelText, x, y - 10);
        
        // Scanning effect for box
        ctx.fillStyle = `rgba(${hexToRgb(color)}, 0.05)`;
        ctx.fillRect(x, y, w, h);
    });
}

function updateBiometrics(results, latency) {
    document.getElementById('inf-val').innerText = `${latency}MS`;
    if (results.length === 0) return;

    const probs = results[0].probabilities;
    let html = '';
    emotions.forEach(emo => {
        const val = (probs[emo] * 100).toFixed(1);
        html += `
            <div class="metric-row">
                <div class="metric-header"><span>${emo.toUpperCase()}</span><span>${val}%</span></div>
                <div class="metric-bar-bg">
                    <div class="metric-bar-fill" style="width: ${val}%; background: ${getEmotionColor(emo)}"></div>
                </div>
            </div>
        `;
    });
    metricContainer.innerHTML = html;
}

function getEmotionColor(emotion) {
    const colors = {
        "Happy": "#39ff14",
        "Sad": "#00f2ff",
        "Angry": "#ff2d55",
        "Surprise": "#ffff44",
        "Neutral": "#7000ff",
        "Fear": "#ff8844",
        "Disgust": "#ff00ff"
    };
    return colors[emotion] || "#00f2ff";
}

function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? `${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(result[3], 16)}` : '0, 242, 255';
}

// 4. Interface Controls
toggleBtn.addEventListener('click', () => {
    if (isStreaming) {
        stream.getTracks().forEach(track => track.stop());
        isStreaming = false;
        toggleBtn.innerText = "Activate Stream";
        toggleBtn.classList.remove('active-btn');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    } else {
        startWebcam();
        toggleBtn.innerText = "Deactivate Stream";
        toggleBtn.classList.add('active-btn');
    }
});

captureBtn.addEventListener('click', () => {
    const link = document.createElement('a');
    link.download = `FaceSentrix_Scan_${Date.now()}.jpg`;
    link.href = canvas.toDataURL('image/jpeg');
    link.click();
});

startWebcam();
