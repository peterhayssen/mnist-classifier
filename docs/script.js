const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let painting = false;

// Event listeners for drawing
canvas.addEventListener('mousedown', startPosition);
canvas.addEventListener('mouseup', endPosition);
canvas.addEventListener('mousemove', draw);

// Function to start drawing
function startPosition(e) {
    painting = true;
    draw(e);
}

// Function to stop drawing
function endPosition() {
    painting = false;
    ctx.beginPath();
}

// Function to draw on the canvas
function draw(e) {
    if (!painting) return;
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
}

// Clear the canvas
function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// Classify the drawn image
function classifyImage() {
    // Convert the canvas drawing to a base64 image
    const imgData = canvas.toDataURL('image/png');
    
    // Send the image to the backend for classification
    fetch('https://temp-address/classify', {  // Replace with your backend URL
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: imgData })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').textContent = 'Predicted number: ' + data.result;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

