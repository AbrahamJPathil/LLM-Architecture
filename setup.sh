#!/bin/bash

# Setup script for WarmStart (Linux/macOS)

echo -e "\033[0;36m🚀 WarmStart Setup\033[0m"
echo ""

# Install system dependencies
echo -e "\033[0;33mChecking system dependencies...\033[0m"

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    echo -e "\033[0;32m✓ Detected OS: $PRETTY_NAME\033[0m"
elif [ "$(uname)" == "Darwin" ]; then
    OS="macos"
    echo -e "\033[0;32m✓ Detected OS: macOS\033[0m"
else
    OS="unknown"
    echo -e "\033[0;33m⚠ Could not detect OS, skipping package installation\033[0m"
fi

# Install required packages based on OS
if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
    echo -e "\033[0;33mInstalling build dependencies for Ubuntu/Debian...\033[0m"
    
    # Check if running with sudo
    if [ "$EUID" -eq 0 ]; then
        apt-get update
        apt-get install -y python3-dev python3-venv build-essential
    else
        echo -e "\033[0;33mRequires sudo to install packages. Please enter password:\033[0m"
        sudo apt-get update
        sudo apt-get install -y python3-dev python3-venv build-essential
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "\033[0;32m✓ System dependencies installed\033[0m"
    else
        echo -e "\033[0;31m✗ Failed to install system dependencies\033[0m"
        echo -e "\033[0;33m  You can continue, but ChromaDB may not install properly\033[0m"
    fi
    
elif [ "$OS" = "fedora" ] || [ "$OS" = "rhel" ] || [ "$OS" = "centos" ]; then
    echo -e "\033[0;33mInstalling build dependencies for Fedora/RHEL/CentOS...\033[0m"
    
    if [ "$EUID" -eq 0 ]; then
        yum install -y python3-devel gcc gcc-c++
    else
        echo -e "\033[0;33mRequires sudo to install packages. Please enter password:\033[0m"
        sudo yum install -y python3-devel gcc gcc-c++
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "\033[0;32m✓ System dependencies installed\033[0m"
    else
        echo -e "\033[0;31m✗ Failed to install system dependencies\033[0m"
        echo -e "\033[0;33m  You can continue, but ChromaDB may not install properly\033[0m"
    fi

elif [ "$OS" = "arch" ] || [ "$OS" = "manjaro" ]; then
    echo -e "\033[0;33mInstalling build dependencies for Arch Linux...\033[0m"
    
    if [ "$EUID" -eq 0 ]; then
        pacman -S --noconfirm python base-devel
    else
        echo -e "\033[0;33mRequires sudo to install packages. Please enter password:\033[0m"
        sudo pacman -S --noconfirm python base-devel
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "\033[0;32m✓ System dependencies installed\033[0m"
    else
        echo -e "\033[0;31m✗ Failed to install system dependencies\033[0m"
        echo -e "\033[0;33m  You can continue, but ChromaDB may not install properly\033[0m"
    fi
    
elif [ "$OS" = "macos" ]; then
    echo -e "\033[0;33mChecking for Xcode Command Line Tools...\033[0m"
    
    if xcode-select -p &> /dev/null; then
        echo -e "\033[0;32m✓ Xcode Command Line Tools already installed\033[0m"
    else
        echo -e "\033[0;33mInstalling Xcode Command Line Tools...\033[0m"
        xcode-select --install
        echo -e "\033[0;33m  Please complete the Xcode installation and run this script again\033[0m"
        exit 0
    fi
else
    echo -e "\033[0;33m⚠ Unknown OS, skipping package installation\033[0m"
    echo -e "\033[0;33m  If ChromaDB fails to install, you may need to manually install:\033[0m"
    echo -e "\033[0;33m  - python3-dev (or python3-devel)\033[0m"
    echo -e "\033[0;33m  - build-essential (or gcc/g++)\033[0m"
fi

echo ""

# Check Python version
echo -e "\033[0;33mChecking Python version...\033[0m"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo -e "\033[0;32m✓ $PYTHON_VERSION\033[0m"
    
    # Extract version number and warn if >= 3.13
    VERSION_NUM=$(echo $PYTHON_VERSION | grep -oP '\d+\.\d+' | head -1)
    MAJOR=$(echo $VERSION_NUM | cut -d. -f1)
    MINOR=$(echo $VERSION_NUM | cut -d. -f2)
    
    if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 13 ]; then
        echo -e "\033[0;33m⚠ Detected Python $MAJOR.$MINOR. Some vector DB dependencies may be unavailable: Chroma and NumPy wheels.\033[0m"
        echo -e "\033[0;33m  Recommended: install Python 3.10-3.12 and create a venv to enable Chroma vector indexing.\033[0m"
    fi
    
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo -e "\033[0;32m✓ $PYTHON_VERSION\033[0m"
    PYTHON_CMD=python
else
    echo -e "\033[0;31m✗ Python not found. Please install Python 3.9 or higher.\033[0m"
    exit 1
fi

# Create virtual environment (prefer uv if available)
echo ""
echo -e "\033[0;33mCreating virtual environment...\033[0m"

if command -v uv &> /dev/null; then
    echo -e "\033[0;32mFound 'uv' - using uv to manage venv and deps\033[0m"
    
    if [ -d "venv" ]; then
        echo -e "\033[0;32m✓ Virtual environment already exists\033[0m"
    else
        echo -e "\033[0;33mCreating venv with Python 3.11 (recommended for vector DB)...\033[0m"
        uv venv --python 3.11 venv
        
        if [ $? -ne 0 ]; then
            echo -e "\033[0;33muv could not create venv with Python 3.11. Trying default interpreter...\033[0m"
            uv venv venv
        fi
        
        echo -e "\033[0;32m✓ Virtual environment created\033[0m"
    fi
    
    echo ""
    echo -e "\033[0;33mActivating virtual environment...\033[0m"
    source venv/bin/activate
    
    echo ""
    echo -e "\033[0;33mInstalling dependencies via uv...\033[0m"
    uv pip install --upgrade pip setuptools wheel
    uv pip install -r requirements.txt
else
    if [ -d "venv" ]; then
        echo -e "\033[0;32m✓ Virtual environment already exists\033[0m"
    else
        $PYTHON_CMD -m venv venv
        if [ $? -eq 0 ]; then
            echo -e "\033[0;32m✓ Virtual environment created\033[0m"
        else
            echo -e "\033[0;31m✗ Failed to create virtual environment\033[0m"
            exit 1
        fi
    fi
    
    # Activate virtual environment
    echo ""
    echo -e "\033[0;33mActivating virtual environment...\033[0m"
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        echo -e "\033[0;31m✗ Virtual environment activation script not found\033[0m"
        exit 1
    fi
    
    # Install dependencies
    echo ""
    echo -e "\033[0;33mInstalling dependencies...\033[0m"
    pip install --upgrade pip
    pip install -r requirements.txt
fi

if [ $? -eq 0 ]; then
    echo -e "\033[0;32m✓ Dependencies installed\033[0m"
else
    echo -e "\033[0;31m✗ Failed to install dependencies\033[0m"
    exit 1
fi

# Create .env file if it doesn't exist
echo ""
echo -e "\033[0;33mSetting up environment variables...\033[0m"
if [ -f ".env" ]; then
    echo -e "\033[0;32m✓ .env file already exists\033[0m"
else
    cp .env.example .env
    echo -e "\033[0;32m✓ Created .env file from template\033[0m"
    echo -e "\033[0;33m  ⚠  Please edit .env and add your API keys!\033[0m"
fi

# Create necessary directories
echo ""
echo -e "\033[0;33mCreating directories...\033[0m"

DIRECTORIES=(
    "data/golden"
    "data/synthetic"
    "data/artifacts"
    "experiments"
    "logs"
)

for dir in "${DIRECTORIES[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "\033[0;32m✓ Created $dir\033[0m"
    fi
done

# Initialize database
echo ""
echo -e "\033[0;33mInitializing database...\033[0m"
python -c 'from src.models.database import init_database; init_database()'

if [ $? -eq 0 ]; then
    echo -e "\033[0;32m✓ Database initialized\033[0m"
else
    echo -e "\033[0;31m✗ Failed to initialize database\033[0m"
fi

echo ""
echo -e "\033[0;32m✅ Setup complete!\033[0m"
echo ""
echo -e "\033[0;36mNext steps:\033[0m"
echo -e "\033[1;37m1. Edit .env and add your API keys\033[0m"
echo -e "\033[1;37m2. Run: python simple_cli.py --help\033[0m"
echo ""
echo -e "\033[0;33mTo activate the virtual environment in the future:\033[0m"
echo -e "\033[1;37m   source venv/bin/activate\033[0m"
echo ""
