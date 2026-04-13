# MacroIntelligence — Deployment Guide (Google Cloud VM + Git)

## 1. One-time Local Setup (Git)

```bash
# Inside your project folder
cd macro_intelligence_complete

# Configure git identity (if not already done)
git config --global user.email "you@example.com"
git config --global user.name "Your Name"

# Create GitHub repo first at https://github.com/new
# Then link it:
git remote add origin https://github.com/YOUR_USERNAME/macro_intelligence_complete.git
# OR if remote already exists:
git remote set-url origin https://github.com/YOUR_USERNAME/macro_intelligence_complete.git

# Stage all fixes
git add -A

# Commit
git commit -m "fix: BOM/CRLF across all .py files, trades=0 dashboard, Disconnected status"

# Push
git push -u origin main
```

---

## 2. Create Google Cloud VM (run once from local machine)

```bash
# Install gcloud CLI if needed: https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Create the VM
gcloud compute instances create macro-intelligence \
  --zone=us-central1-a \
  --machine-type=e2-standard-2 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=30GB \
  --tags=http-server,https-server

# Open dashboard port
gcloud compute firewall-rules create allow-macro-dashboard \
  --allow=tcp:8888 \
  --target-tags=http-server \
  --description="MacroIntel Dashboard port"
```

---

## 3. First-Time VM Setup

```bash
# SSH into VM
gcloud compute ssh macro-intelligence --zone=us-central1-a

# Install system packages
sudo apt update && sudo apt install -y \
  python3 python3-pip python3-venv git screen curl

# Clone your repo
git clone https://github.com/YOUR_USERNAME/macro_intelligence_complete.git
cd macro_intelligence_complete

# Copy your .env from local machine (run this on your LOCAL terminal, not VM)
gcloud compute scp project/.env macro-intelligence:~/macro_intelligence_complete/project/.env \
  --zone=us-central1-a

# Back on VM — start the dashboard
chmod +x start.sh stop.sh
./start.sh
```

---

## 4. Daily Git Workflow (Push from Local → Pull on VM)

### On your LOCAL machine (after making changes):
```bash
cd macro_intelligence_complete
git add -A
git commit -m "your change description"
git push origin main
```

### On the VM (to pull latest and restart):
```bash
cd ~/macro_intelligence_complete

# Pull latest code
git pull origin main

# Restart dashboard
./stop.sh
./start.sh
```

---

## 5. Useful VM Commands

```bash
# View live logs
tail -f ~/macro_intelligence_complete/dashboard.log

# Attach to running session
screen -r macro

# Detach from screen (without stopping)
# Press: Ctrl+A then D

# Check if running
screen -list

# Stop dashboard
./stop.sh

# Check VM external IP
curl ifconfig.me

# Access dashboard
# http://YOUR_VM_IP:8888
```

---

## 6. Auto-restart on VM Reboot (optional)

```bash
# Add to crontab on VM
crontab -e

# Add this line:
@reboot sleep 30 && /home/$USER/macro_intelligence_complete/start.sh >> /home/$USER/macro_intelligence_complete/cron.log 2>&1
```

---

## Bug Fixes Applied in This Commit

| # | Issue | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | Dashboard not loading | UTF-8 BOM + mixed CRLF/LF line endings in all `.py` files caused Python import to fail with `SyntaxError: invalid non-printable character U+FEFF` | Removed BOM and normalized to LF across all 33 Python files |
| 2 | Trades = 0 in all 3 domains | Push loop never emitted trade data via WebSocket; `reportData` was `null` on initial render so `laneReportMap()` returned empty | Added `daily_report_update` WebSocket emit every 15s in push loop; wired JS listener `io_sock.on('daily_report_update', renderDailyReport)` |
| 3 | "Disconnected" status stuck | HTML hardcoded red "Disconnected" on load; only changed on socket `connect` event which could be slow/blocked | Changed initial state to amber "Connecting..."; added 8s fallback timeout that sets "Polling" if socket never fires |

