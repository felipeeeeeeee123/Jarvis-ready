#!/bin/bash
python3 backend/server.py &
python3 keep_alive.py
