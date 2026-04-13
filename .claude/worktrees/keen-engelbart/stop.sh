#!/bin/bash
screen -S macro -X quit 2>/dev/null && echo "Stopped." || echo "Not running."
