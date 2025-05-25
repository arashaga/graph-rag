/**
 * Run this file to debug WebSocket connections:
 * node debug.js
 */
const { WebSocket } = require('ws');
const ws = new WebSocket('ws://localhost:5173');

ws.on('open', function open() {
  console.log('WebSocket connection established');
  ws.send('test');
});

ws.on('message', function incoming(data) {
  console.log('Message received:', data);
});

ws.on('error', function error(err) {
  console.error('WebSocket error:', err);
});

ws.on('close', function close() {
  console.log('WebSocket connection closed');
});
