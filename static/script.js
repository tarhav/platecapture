document.addEventListener('DOMContentLoaded', function() {
    const videoElement = document.getElementById('videoElement');
    const videoCanvas = document.getElementById('videoCanvas');
    const plateList = document.getElementById('plateList');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const enableCameraBtn = document.getElementById('enableCameraBtn');
    const cameraError = document.getElementById('cameraError');
    const videoCtx = videoCanvas.getContext('2d');

    let currentPlate = {
        text: null,
        rect: null,
        timestamp: 0
    };
    const detectedPlates = new Set();
    let isProcessing = false;

    // Check camera permission state
    async function checkCameraPermission() {
        try {
            const permissions = await navigator.permissions.query({ name: 'camera' });
            if (permissions.state === 'denied') {
                showCameraButton();
                return false;
            }
            return true;
        } catch (error) {
            console.error('Error checking camera permission:', error);
            return false;
        }
    }

    // Show camera enable button
    function showCameraButton() {
        enableCameraBtn.classList.remove('hidden');
        cameraError.classList.remove('hidden');
        loadingIndicator.querySelector('.animate-spin').classList.add('hidden');
        loadingIndicator.querySelector('p').textContent = 'Camera access required';
    }

    // Handle camera enable button click
    enableCameraBtn.addEventListener('click', async () => {
        try {
            await setupCamera();
            enableCameraBtn.classList.add('hidden');
            cameraError.classList.add('hidden');
            loadingIndicator.classList.add('hidden');
        } catch (error) {
            console.error('Error enabling camera:', error);
            cameraError.textContent = 'Failed to enable camera. Please ensure permission is granted.';
            cameraError.classList.remove('hidden');
        }
    });

    async function setupCamera() {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        });
        videoElement.srcObject = stream;

        return new Promise(resolve => {
            videoElement.onloadedmetadata = () => {
                videoElement.play();
                videoCanvas.width = videoElement.videoWidth;
                videoCanvas.height = videoElement.videoHeight;
                loadingIndicator.classList.add('hidden');
                resolve();
            };
        });
    }

    // Detect plate region (made with Claude-Sonnet-3.5 and GPT-4o)
    function detectLicensePlateRegion(src) {
        let gray = new cv.Mat();
        let edges = new cv.Mat();
        let contours = new cv.MatVector();
        let hierarchy = new cv.Mat();

        // Preprocessing pipeline
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
        cv.GaussianBlur(gray, gray, new cv.Size(5, 5), 0);
        cv.equalizeHist(gray, gray);

        // Edge detection
        cv.Canny(gray, edges, 50, 150);
        cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        let licensePlateContour = null;
        for (let i = 0; i < contours.size(); i++) {
            let contour = contours.get(i);
            let area = cv.contourArea(contour);
            let approx = new cv.Mat();
            cv.approxPolyDP(contour, approx, 0.02 * cv.arcLength(contour, true), true);

            if (approx.rows === 4 && area > 500 && area < 10000) {
                licensePlateContour = approx;
                break;
            }
            approx.delete();
        }

        // Cleanup
        gray.delete();
        edges.delete();
        contours.delete();
        hierarchy.delete();

        return licensePlateContour;
    }

    async function processFrame() {
        if (isProcessing) return;
        isProcessing = true;

        try {
            videoCtx.drawImage(videoElement, 0, 0);
            let src = cv.imread(videoCanvas);

            let contour = detectLicensePlateRegion(src);
            if (contour) {
                let rect = cv.boundingRect(contour);
                let aspectRatio = rect.width / rect.height;

                if (aspectRatio >= 2 && aspectRatio <= 5) {
                    currentPlate.rect = rect;
                    currentPlate.timestamp = Date.now();

                    // OCR Processing
                    let roi = src.roi(rect);
                    let processed = new cv.Mat();
                    cv.cvtColor(roi, processed, cv.COLOR_RGBA2GRAY);
                    cv.adaptiveThreshold(processed, processed, 255,
                        cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2);

                    let canvas = document.createElement('canvas');
                    cv.imshow(canvas, processed);

                    const { data: { text } } = await Tesseract.recognize(canvas, 'eng', {
                        tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                        tessedit_pageseg_mode: 'single_line'
                    });

                    const cleanedText = text.replace(/[^A-Z0-9]/g, '').trim();
                    if (/^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$/.test(cleanedText)) {
                        currentPlate.text = cleanedText;
                        if (!detectedPlates.has(cleanedText)) {
                            addToPlateList(cleanedText);
                            detectedPlates.add(cleanedText);
                        }
                    }

                    processed.delete();
                    roi.delete();
                }
            }
            src.delete();
        } catch (error) {
            console.error('Processing error:', error);
        }
        isProcessing = false;
    }

    // Draw bounding box (made with DeepSeek-V3 and Claude-Sonnet-3.5)
    function drawBoundingBox() {
        videoCtx.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
        videoCtx.drawImage(videoElement, 0, 0);

        if (currentPlate.rect && (Date.now() - currentPlate.timestamp) < 1000) {
            const rect = currentPlate.rect;

            // Draw bounding box
            videoCtx.beginPath();
            videoCtx.rect(rect.x, rect.y, rect.width, rect.height);
            videoCtx.lineWidth = 3;
            videoCtx.strokeStyle = '#3b82f6';
            videoCtx.stroke();

            // Draw text with background
            const text = currentPlate.text || 'Detecting...';
            videoCtx.font = 'bold 24px Arial';
            videoCtx.textBaseline = 'bottom';

            // Text background
            const textWidth = videoCtx.measureText(text).width;
            videoCtx.fillStyle = 'rgba(30, 41, 59, 0.9)';
            videoCtx.fillRect(
                rect.x - 5,
                rect.y - 34,
                textWidth + 10,
                30
            );

            // Text
            videoCtx.fillStyle = '#3b82f6';
            videoCtx.fillText(text, rect.x, rect.y - 10);
        }

        requestAnimationFrame(drawBoundingBox);
    }

    // Add plate items to list (made with DeepSeek-V3)
    function addToPlateList(plateNumber) {
        // Show the "Detected Plates" section if it's hidden
        const detectedPlatesSection = document.getElementById('detectedPlatesSection');
        if (detectedPlatesSection.style.display === 'none') {
            detectedPlatesSection.style.display = 'block';
        }

        const listItem = document.createElement('li');
        listItem.className = 'bg-gray-800 p-4 rounded-lg flex items-center justify-between border border-gray-700 animate-slide-down';
        listItem.innerHTML = `
            <span class="text-blue-400 font-mono">${plateNumber}</span>
            <span class="text-gray-400 text-sm">${new Date().toLocaleTimeString()}</span>
        `;

        // Add new item at the top
        if (plateList.firstChild) {
            plateList.insertBefore(listItem, plateList.firstChild);
        } else {
            plateList.appendChild(listItem);
        }

        // Remove oldest item if more than 10
        if (plateList.children.length > 10) {
            plateList.removeChild(plateList.lastChild);
        }
    }

    async function main() {
        const hasPermission = await checkCameraPermission();
        if (!hasPermission) return;

        await Promise.all([
            new Promise(resolve => cv.onRuntimeInitialized = resolve),
            Tesseract.ready
        ]);

        try {
            await setupCamera();
            requestAnimationFrame(drawBoundingBox);
            setInterval(processFrame, 100);
        } catch (error) {
            console.error('Error initializing camera:', error);
            showCameraButton();
        }
    }

    // Handle mobile orientation changes
    window.addEventListener('resize', () => {
        videoCanvas.width = videoElement.videoWidth;
        videoCanvas.height = videoElement.videoHeight;
    });

    main().catch(console.error);
});
