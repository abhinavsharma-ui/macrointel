// WEBSOCKET RECONNECTION FIX - JavaScript Version
// Apply this if patch_websocket.py doesn't work
// 
// Usage:
// 1. Find your socket.io client initialization in your HTML/JS
// 2. Replace or add this configuration
// 3. Test by opening browser console: look for "Socket connected" without errors
//
// Before (BROKEN):
// var socket = io();
//
// After (FIXED):
// var socket = io(undefined, {
//   reconnectionAttempts: Infinity,
//   reconnectionDelay: 1000,
//   reconnectionDelayMax: 5000,
//   transports: ['websocket', 'polling']
// });

const socketIOConfig = {
  reconnectionAttempts: Infinity,           // Never stop trying
  reconnectionDelay: 1000,                  // Initial delay: 1 second
  reconnectionDelayMax: 5000,               // Max delay: 5 seconds (exponential backoff)
  transports: ['websocket', 'polling'],    // Try websocket first, fallback to polling
  upgrade: true,
  forceNew: false,
  path: '/socket.io/',
  query: {
    t: new Date().getTime()                // Cache busting
  }
};

// Initialize socket with config
const socket = io(undefined, socketIOConfig);

// Optional: Add event listeners for debugging
socket.on('connect', () => {
  console.log('[Socket.IO] ✅ Connected');
  console.log('  Socket ID:', socket.id);
  console.log('  Transport:', socket.io.engine.transport.name);
});

socket.on('disconnect', (reason) => {
  console.log('[Socket.IO] ❌ Disconnected:', reason);
  if (reason === 'io server disconnect') {
    // The server has forcefully disconnected the socket with socket.disconnect()
    socket.connect();
  } else if (reason === 'io client namespace disconnect') {
    // The socket has been disconnected with socket.disconnect()
  }
  // else the socket will automatically attempt to reconnect
});

socket.on('connect_error', (error) => {
  console.log('[Socket.IO] ⚠️ Connection Error:', error.message);
});

socket.on('reconnect', () => {
  console.log('[Socket.IO] 🔄 Reconnected after disconnect');
});

socket.on('reconnect_attempt', () => {
  console.log('[Socket.IO] 🔄 Attempting to reconnect...');
});

socket.on('reconnect_failed', () => {
  console.log('[Socket.IO] ❌ Reconnection failed - checking server');
});

// Optional: Check connection status periodically
setInterval(() => {
  if (!socket.connected) {
    console.log('[Socket.IO] Status: Attempting reconnection...');
  }
}, 5000);

// =============================================================================
// IMPLEMENTATION STEPS
// =============================================================================
//
// 1. Find your socket initialization in your HTML/Flask template
//    (usually in <script> tag or external JS file)
//
// 2. Look for: var socket = io();
//              or: const socket = io();
//              or: socket = io.connect();
//
// 3. Replace with the socketIOConfig above and socket initialization
//
// 4. Test:
//    a. Open your app in browser
//    b. Open DevTools (F12 or Cmd+Opt+I)
//    c. Go to Console tab
//    d. You should see: "[Socket.IO] ✅ Connected"
//    e. Close laptop lid for 3+ hours
//    f. Wake up - should see "[Socket.IO] 🔄 Reconnected after disconnect"
//
// 5. Check the Dashboard:
//    - Should NOT show "Socket disconnected" message anymore
//    - Should automatically update when server has new data
//
// =============================================================================
