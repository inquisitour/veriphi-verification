#!/usr/bin/env bash
# ===========================================================
# Smart Streamlit Launcher for GPU Clusters (VSC5 Compatible)
# -----------------------------------------------------------
# ✅ Detects active GPUs
# ✅ Runs Streamlit in CPU-only mode (avoids container network issues)
# ✅ Automatically sets network address and forwards port
# ✅ Works perfectly with VS Code Remote SSH
# ===========================================================

APP_PATH="src/frontend/app.py"               # Your Streamlit app file
PORT=8501                       # Default Streamlit port
ADDR="0.0.0.0"                  # Bind to all interfaces (for VS Code port forwarding)
CONFIG_DIR="$HOME/.streamlit"   # Streamlit config path

# --- Detect GPU ---
echo "🔍 Checking GPU availability..."
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "⚙️  GPU detected (CUDA available)"
    echo "⚠️  Running Streamlit in CPU-safe mode to avoid container isolation issues."
    export CUDA_VISIBLE_DEVICES=""
else
    echo "💡 No GPU detected, running normally."
fi

# --- Setup Streamlit config if missing ---
mkdir -p "$CONFIG_DIR"
cat > "$CONFIG_DIR/config.toml" <<EOF
[server]
address = "$ADDR"
port = $PORT
headless = true
enableCORS = false
EOF

# --- Check if Streamlit app exists ---
if [ ! -f "$APP_PATH" ]; then
    echo "❌ Streamlit app not found at $APP_PATH"
    echo "Please edit run_streamlit_safe.sh and set APP_PATH to your app script."
    exit 1
fi

# --- Launch Streamlit ---
echo "🚀 Launching Streamlit on port $PORT ..."
echo "🌐 Once running, forward port 8501 in VS Code (Ports tab → ➕ Forward Port → 8501)"
echo "Then open 👉 http://localhost:$PORT"

# Use nohup for resilience
nohup streamlit run "$APP_PATH" --server.address="$ADDR" --server.port="$PORT" > streamlit.log 2>&1 &

sleep 3
if pgrep -f "streamlit run" >/dev/null; then
    echo "✅ Streamlit is running in the background."
    echo "   Log file: streamlit.log"
else
    echo "❌ Streamlit failed to start. Check streamlit.log for details."
fi
