const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Access camera
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    video.srcObject = stream;
});

video.addEventListener('play', () => {
    const sendFrame = async () => {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg');

        try {
            const response = await fetch('/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataUrl })
            });
            const result = await response.json();
            if (result.image) {
                const img = new Image();
                img.src = result.image;
                img.onload = () => ctx.drawImage(img, 0, 0);
            }
        } catch (err) {
            console.error(err);
        }
        requestAnimationFrame(sendFrame);
    };
    sendFrame();
});
