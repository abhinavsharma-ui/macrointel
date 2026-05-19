#!/bin/bash
cd /home/abhinavsharma1359/macro_intelligence_complete
while true; do
    if ! pgrep -f run.py > /dev/null; then
        echo "$(date): Process died, restarting..." >> watchdog.log
        nohup venv/bin/python project/run.py >> run_log.txt 2>&1 &
        echo "$(date): Restarted PID $!" >> watchdog.log
    fi
    sleep 30
done
