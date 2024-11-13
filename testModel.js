const http = require('http');

const data = JSON.stringify({
    question: ' Punishment for attack on a person'
});

const options = {
    hostname: 'localhost',
    port: 5000,
    path: '/ask',
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Content-Length': data.length
    }
};

const req = http.request(options, (res) => {
    let responseData = '';

    res.on('data', (chunk) => {
        responseData += chunk;
    });

    res.on('end', () => {
        console.log(JSON.parse(responseData));
    });
});

req.on('error', (error) => {
    console.error('Error:', error);
});

req.write(data);
req.end();