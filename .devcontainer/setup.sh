#!/bin/bash
set -e

echo "🔧 Installing ODBC libraries and Microsoft SQL Server ODBC driver..."

# Set non-interactive mode and persist env
export DEBIAN_FRONTEND=noninteractive

# Install core dependencies
sudo apt-get update
sudo apt-get install -y curl gnupg apt-transport-https ca-certificates unixodbc unixodbc-dev libpq-dev debconf-utils

# Add Microsoft signing key securely
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | sudo tee /usr/share/keyrings/microsoft-prod.gpg > /dev/null

# Register Microsoft's repo (Bookworm/Debian 12)
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" | sudo tee /etc/apt/sources.list.d/mssql-release.list

# Update package lists
sudo apt-get update

# Accept license for msodbcsql18
echo "msodbcsql18 msodbcsql18/accept_eula select true" | sudo -E debconf-set-selections
ACCEPT_EULA=Y sudo -E apt-get install -y msodbcsql18

# ----------------------------------------
# CONDA: Set up Python environment
# ----------------------------------------

echo "🐍 Setting up Conda environment: ai-agents"

# Create Conda environment
conda create -y -n ai-agents python=3.12

# Upgrade pip and install requirements in the conda env
echo "📦 Installing requirements.txt into Conda env..."
conda run -n ai-agents pip install --upgrade pip
conda run -n ai-agents pip install -r requirements.txt
conda run -n ai-agents pip install ipykernel
# Register Jupyter kernel
echo "🧠 Registering Conda env with Jupyter..."
conda run -n ai-agents python -m ipykernel install --user --name ai-agents --display-name "Python (ai-agents)"

# ----------------------------------------
# NODE: Install Vite, React, and TypeScript globally
# ----------------------------------------

# Install Vite, React, ReactDOM, TypeScript, and types globally for scaffolding
npm install -g create-vite@latest typescript @types/react @types/react-dom

echo "✅ Dev container setup complete!"