const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);

// Access camera
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    video.srcObject = stream;
});

video.addEventListener('play', () => {
    const sendFrame = () => {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg');

        socket.emit('frame', dataUrl);
        requestAnimationFrame(sendFrame);
    };
    sendFrame();
});

socket.on('processed_frame', data => {
    const img = new Image();
    img.src = data;
    img.onload = () => ctx.drawImage(img, 0, 0);
});
