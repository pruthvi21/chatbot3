#!/usr/bin/env bash
# Set up a virtualenv, install deps, and launch the Streamlit chatbot.
set -e

cd "$(dirname "$0")"

# 1. Create venv on first run
if [ ! -d ".venv" ]; then
  echo "==> Creating virtual environment (.venv)…"
  python3 -m venv .venv
fi

# 2. Activate it
source .venv/bin/activate

# 3. Install / update dependencies
echo "==> Installing dependencies…"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# 4. Ensure an API key is available (this folder's .env or the project .env)
if [ ! -f ".env" ] && [ ! -f "../.env" ]; then
  echo "!! No .env found. Create memory_chatbot/.env with:"
  echo "   GROQ_API_KEY=gsk_your_key_here"
  exit 1
fi

# 5. Launch
echo "==> Starting Streamlit…  (Ctrl+C to stop)"
exec streamlit run app.py
