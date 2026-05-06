#!/usr/bin/env bash
# ============================================================
# setup.sh – One-shot setup & launch script for MediAssist
# Usage:  chmod +x setup.sh && ./setup.sh
# ============================================================
set -e

BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}${BOLD}"
echo "  ╔══════════════════════════════════════════╗"
echo "  ║   MediAssist – AI Medical Chatbot Setup  ║"
echo "  ╚══════════════════════════════════════════╝"
echo -e "${NC}"

# Check Python
python3 --version >/dev/null 2>&1 || { echo "Python 3.9+ is required."; exit 1; }

# Create .env if missing
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  No .env file found. Copying from .env.example …${NC}"
    cp .env.example .env
    echo -e "${YELLOW}   → Please edit .env and add your GROQ_API_KEY and TAVILY_API_KEY${NC}"
    echo ""
fi

# Virtual environment
if [ ! -d "venv" ]; then
    echo -e "${CYAN}Creating virtual environment …${NC}"
    python3 -m venv venv
fi

source venv/bin/activate

# Install deps
echo -e "${CYAN}Installing dependencies …${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo -e "${GREEN}${BOLD}✅ Setup complete!${NC}"
echo ""
echo -e "${CYAN}Starting MediAssist on http://localhost:8000 …${NC}"
echo -e "   Open ${BOLD}http://localhost:8000${NC} in your browser."
echo ""

python main.py
