#!/bin/bash

announce() {
if hash notify-send 2>/dev/null; then
  notify-send "$1"
else
  echo "$1"
fi
}

announce "Launching Jupyter notebook"
jupyter notebook
