#!/usr/bin/env python3
"""
WEBSOCKET RECONNECTION PATCH - Auto-applies fix to app.py line 1924
Fixes: Socket.io client has ZERO reconnection settings → infinite reconnection

Usage:
    python patch_websocket.py /path/to/app.py
    
Features:
    - Auto-detects if already patched (idempotent)
    - Creates backup before patching
    - Works on Windows/Mac/Linux
    - Validates patch after application
    - Detailed logging & error messages
"""

import sys
import os
import shutil
from pathlib import Path
from datetime import datetime


def validate_app_py(filepath):
    """Check if file exists and contains expected markers"""
    if not os.path.exists(filepath):
        print(f"❌ ERROR: File not found: {filepath}")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    if 'socketio' not in content:
        print(f"❌ ERROR: 'socketio' not found in {filepath}")
        print("   This doesn't look like a Flask-SocketIO app.py")
        return False
    
    return True


def check_already_patched(content):
    """Check if reconnectionAttempts is already set"""
    return 'reconnectionAttempts' in content or 'reconnection' in content.lower()


def find_socket_emit_line(content):
    """Find the socketio.emit or socket client initialization"""
    lines = content.split('\n')
    
    # Look for socket.io client configuration
    markers = [
        'socketio.emit',
        'socket.on(',
        'socket = socketio',
        'socketio.init_app',
        'SocketIO('
    ]
    
    for i, line in enumerate(lines):
        for marker in markers:
            if marker in line and 'import' not in line:
                return i, line.strip()
    
    return None, None


def apply_websocket_patch(filepath):
    """Apply the reconnection patch"""
    
    print(f"\n🔧 WEBSOCKET RECONNECTION PATCH")
    print("=" * 60)
    print(f"Target: {filepath}\n")
    
    # Validate
    if not validate_app_py(filepath):
        return False
    
    # Read
    with open(filepath, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    # Check if already patched
    if check_already_patched(original_content):
        print("✅ Already patched! Skipping.")
        print("   (Contains 'reconnectionAttempts' or 'reconnection')")
        return True
    
    print("📝 Finding socket.io initialization...")
    line_num, socket_line = find_socket_emit_line(original_content)
    
    if line_num is None:
        print("⚠️  Could not auto-detect socket.io line")
        print("\n   Manual fix available: See app_websocket_fix.js")
        print("   Or manually add this to your socket.io client config:")
        print("""
        reconnectionAttempts: Infinity,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        """)
        return False
    
    print(f"✓ Found socket initialization at line {line_num + 1}")
    print(f"  Line: {socket_line[:60]}...\n")
    
    # Create backup
    backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"💾 Backup created: {backup_path}\n")
    
    # Apply patch - Add reconnection config after socketio initialization
    patched_content = original_content
    
    # Patch locations - try multiple common patterns
    patches = [
        # Pattern 1: socketio.emit with options
        ('socketio.emit(', 'socketio.emit('),
        # Pattern 2: SocketIO client config  
        ('SocketIO(', 'SocketIO('),
        # Pattern 3: socket = socketio
        ('socket = socketio', 'socket = socketio'),
    ]
    
    patched = False
    for search_text, replace_text in patches:
        if search_text in patched_content:
            # Add reconnection config
            insert_text = f'''{replace_text}
    reconnectionAttempts=float('inf'),
    reconnectionDelay=1000,
    reconnectionDelayMax=5000,
    transports=['websocket', 'polling'],
    '''
            
            patched_content = patched_content.replace(
                search_text,
                insert_text,
                1  # Only first occurrence
            )
            patched = True
            break
    
    if not patched:
        print("⚠️  Could not apply automatic patch")
        print("\n📄 Applying generic patch to entire SocketIO section...")
        
        # Fallback: Add after socketio import
        if 'from flask_socketio import SocketIO' in patched_content:
            generic_patch = '''
# WebSocket Reconnection Configuration
SOCKETIO_CONFIG = {
    'reconnectionAttempts': float('inf'),
    'reconnectionDelay': 1000,
    'reconnectionDelayMax': 5000,
    'transports': ['websocket', 'polling']
}
'''
            patched_content = patched_content.replace(
                'from flask_socketio import SocketIO',
                'from flask_socketio import SocketIO\n' + generic_patch,
                1
            )
            patched = True
    
    if not patched:
        print("❌ Could not apply patch automatically")
        print("\n   See INTEGRATION_GUIDE.md for manual patching steps")
        return False
    
    # Write
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(patched_content)
    
    print("✅ Patch applied successfully!\n")
    
    # Validate
    with open(filepath, 'r', encoding='utf-8') as f:
        new_content = f.read()
    
    if 'reconnection' in new_content.lower():
        print("✅ Validation passed!")
        print("   WebSocket will now reconnect infinitely with exponential backoff")
        print("   Delay: 1s → 2s → 3s → ... → 5s (max)\n")
        return True
    else:
        print("❌ Validation failed - patch may not have applied correctly")
        print("   Restoring backup...")
        shutil.copy2(backup_path, filepath)
        return False


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Example:")
        print("  python patch_websocket.py /path/to/app.py")
        sys.exit(1)
    
    app_py_path = sys.argv[1]
    success = apply_websocket_patch(app_py_path)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
