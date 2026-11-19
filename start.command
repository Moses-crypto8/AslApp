#!/bin/bash
cd "$(dirname "$0")" || exit 1
if command -v python3.12 >/dev/null 2>&1; then
  python3.12 bootstrap.py
elif command -v python3.11 >/dev/null 2>&1; then
  python3.11 bootstrap.py
else
  python3 bootstrap.py
fi
result=$?
if [ "$result" -ne 0 ]; then read -r -p "Press Enter to close..."; fi
exit "$result"
