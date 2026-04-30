/**
 * FaceSentrix - Frontend Logic
 */

const video = document.getElementById('webcam-feed');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const probContainer = document.getElementById('prob-container');
const toggleBtn = document.getElementById('toggle-cam');

const emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"];
let isStreaming = false;
let stream = null;

// 1. Initialize Video Stream
async function startWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        video.srcObject = stream;
        isStreaming = true;
        
        // Match canvas size to video once loaded
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            document.getElementById('res-val').innerText = `${video.videoWidth}x${video.videoHeight}`;
            processFrame(); // Start pipeline
        };
    } catch (err) {
        console.error("Error accessing webcam:", err);
        alert("Could not access webcam. Please ensure permissions are granted.");
    }
}

// 2. Main Processing Loop
async function processFrame() {
    if (!isStreaming) return;

    // Capture frame from video
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0);

    const base64Image = tempCanvas.toDataURL('image/jpeg', 0.7);

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
            drawOverlays(data.results);
            updateStats(data.results, (endTime - startTime).toFixed(0));
        }
    } catch (err) {
        console.error("Prediction error:", err);
    }

    // Schedule next frame (approx 5-10 FPS for stability)
    setTimeout(processFrame, 100);
}

// 3. Visualization
function drawOverlays(results) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    document.getElementById('face-count').innerText = results.length;

    results.forEach(res => {
        const [x, y, w, h] = res.box;
        
        // Draw Box
        ctx.strokeStyle = '#00d2ff';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        // Draw Label Background
        ctx.fillStyle = '#00d2ff';
        ctx.font = 'bold 16px Inter';
        const labelText = `${res.emotion} (${(res.confidence * 100).toFixed(0)}%)`;
        const textWidth = ctx.measureText(labelText).width;
        ctx.fillRect(x, y - 25, textWidth + 10, 25);

        // Draw Text
        ctx.fillStyle = '#fff';
        ctx.fillText(labelText, x + 5, y - 7);
    });
}

function updateStats(results, inferenceTime) {
    document.getElementById('inf-val').innerText = `${inferenceTime}ms`;
    
    if (results.length === 0) return;

    // Use the first detected face for the side-panel bars
    const mainFace = results[0];
    const probs = mainFace.probabilities;

    let html = '';
    emotions.forEach(emo => {
        const val = (probs[emo] * 100).toFixed(1);
        html += `
            <div class="prob-item">
                <div class="prob-info"><span>${emo}</span><span>${val}%</span></div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${val}%; background: ${getEmotionColor(emo)}"></div>
                </div>
            </div>
        `;
    });
    probContainer.innerHTML = html;
}

function getEmotionColor(emotion) {
    const colors = {
        "Happy": "#39ff14",
        "Sad": "#00d2ff",
        "Angry": "#ff4444",
        "Surprise": "#ffff44",
        "Neutral": "#a1a1a6",
        "Fear": "#ff8844",
        "Disgust": "#aa44ff"
    };
    return colors[emotion] || "#fff";
}

// 4. Control Event Listeners
toggleBtn.addEventListener('click', () => {
    if (isStreaming) {
        stream.getTracks().forEach(track => track.stop());
        isStreaming = false;
        toggleBtn.innerText = "Start Camera";
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    } else {
        startWebcam();
        toggleBtn.innerText = "Stop Camera";
    }
});

// Auto-start
startWebcam();
