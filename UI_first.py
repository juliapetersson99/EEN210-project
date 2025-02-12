<!DOCTYPE html>
<html>

<head>
    <title>Fall Detection - Real-time Data</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        #messages {
            overflow-y: scroll;
            height: 90vh;
            width: 60vw;
        }
        canvas {
            width: 100% !important;
            height: 400px !important;
        }
    </style>
</head>

<body>
    <h1>Real-time Data Collection</h1>
    <button id="closeButton">Close Connection</button>
    <div id="buttons"></div>
    <div id="current_state"></div>
    <ul id="messages"></ul>

    <canvas id="dataChart"></canvas>

    <script>
        var ws = new WebSocket("ws://172.20.10.3:8000/ws");
        var labelsArray = [];
        var dataArray = [];
        var maxDataPoints = 50;

        var ctx = document.getElementById('dataChart').getContext('2d');
        var chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labelsArray,
                datasets: [{
                    label: 'Real-time Data',
                    data: dataArray,
                    borderColor: 'blue',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: { 
                        display: true, 
                        title: { display: true, text: 'Time' }
                    },
                    y: { 
                        display: true, 
                        title: { display: true, text: 'Value' }
                    }
                }
            }
        });
ws.onmessage = function (event) {
            console.log("Received data:", event.data);

            var messages = document.getElementById('messages');
            var message = document.createElement('li');
            var content = document.createTextNode(event.data);
            message.appendChild(content);
            messages.appendChild(message);
            messages.scrollTop = messages.scrollHeight;

            try {
                const data = JSON.parse(event.data);
                if (data.value !== undefined) {
                    labelsArray.push(new Date().toLocaleTimeString());
                    dataArray.push(data.value);

                    if (labelsArray.length > maxDataPoints) {
                        labelsArray.shift();
                        dataArray.shift();
                    }

                    chart.update();
                }
                if (data.label) {
                    document.getElementById('current_state').innerHTML = <h2>Current Activity:</h2><h3>${data.label}</h3>;
                }
            } catch (e) {
                console.log("Not a valid JSON message");
            }
        };

        document.getElementById('closeButton').addEventListener('click', function () {
            ws.close();
            console.log("Connection Closed");
        });
    </script>
</body>
</html>