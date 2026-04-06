document.addEventListener("DOMContentLoaded", async function () {

    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const status = document.getElementById("status");

    // ✅ Start camera
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        status.innerText = "Camera Started ✅";
    } catch (err) {
        status.innerText = "Camera Access Denied ❌";
        console.error(err);
        return;
    }

    // ✅ Send frames every 1 second
    setInterval(() => {

        const context = canvas.getContext("2d");
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        const imageData = canvas.toDataURL("image/jpeg");

        fetch("/detect", {
            method: "POST",
            body: JSON.stringify({ image: imageData }),
            headers: {
                "Content-Type": "application/json"
            }
        })
        .then(res => res.json())
        .then(data => {
            console.log("Server Response:", data);
            status.innerText = data.status;
        })
        .catch(err => {
            console.error(err);
            status.innerText = "Connection error ❌";
        });

    }, 1000);

});