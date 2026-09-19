const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const path = require('path')

const app = express();
//обработка json
app.use(express.json());
app.use(express.static(path.join(__dirname, 'frontend')));
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'frontend', 'index.html'));
}); 

//HTTP сервер
const server = http.createServer(app);

//WebSocket на HTTP
const wss = new WebSocket.Server({ server });

wss.on('connection', (ws) => {
    console.log('Новый клиент (браузер) подключился к интерфейсу');
});

//точка для post
app.post('/api/update', (req, res) => {
    const data = req.body;
    
    wss.clients.forEach(client => {
        
        if (client.readyState === WebSocket.OPEN) {
            client.send(JSON.stringify(data));
        }
    });

   
    res.status(200).json({ status: 'success', bytes_sent: JSON.stringify(data).length });
});

//Запуск сервера
const PORT = 3000;
server.listen(PORT, () => {
    console.log(`Сервер интерфейса запущен: http://localhost:${PORT}`);
    console.log(`POST-запросы принимаются на: http://localhost:${PORT}/api/update`);
});