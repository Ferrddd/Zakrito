
const fs = require('fs');
const sampleJson = JSON.parse(fs.readFileSync('./request_example.json', 'utf8'));

fetch('http://localhost:3000/api/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(sampleJson)
})
.then(res => res.json())
.then(data => console.log('Ответ сервера:', data))
.catch(err => console.error('Ошибка отправки:', err));