# Cross-Platform Setup Summary

## ✅ What Was Added

### 1. **setup.sh** - Linux/macOS Setup Script
A complete bash equivalent of `setup.ps1` that:
- ✅ Checks Python version (warns if >= 3.13)
- ✅ Creates virtual environment (prefers `uv` if available)
- ✅ Installs dependencies from `requirements.txt`
- ✅ Creates `.env` from template
- ✅ Creates necessary directories
- ✅ Initializes database
- ✅ Color-coded output for easy reading

**Usage:**
```bash
chmod +x setup.sh
./setup.sh
```

### 2. **Updated GETTING_STARTED.md**
Enhanced with platform-specific instructions:
- ✅ Prerequisites section now covers Windows, Linux, and macOS
- ✅ All setup steps include both PowerShell and Bash commands
- ✅ Platform-specific troubleshooting
- ✅ Linux package installation instructions (apt, yum, brew)
- ✅ File path conventions for each platform (backslash vs forward slash)

### 3. **Updated DOCUMENTATION.md**
Added cross-platform commands in:
- ✅ Installation section
- ✅ Basic usage examples
- ✅ Quick start guide

---

## 📋 Platform Support Matrix

| Feature | Windows | Linux | macOS |
|---------|---------|-------|-------|
| Setup script | `setup.ps1` ✅ | `setup.sh` ✅ | `setup.sh` ✅ |
| Virtual env | `.\venv\Scripts\Activate.ps1` | `source venv/bin/activate` | `source venv/bin/activate` |
| Python command | `python` | `python3` / `python` | `python3` / `python` |
| File paths | Backslash `\` | Forward slash `/` | Forward slash `/` |
| Dependencies | Full support ✅ | Full support ✅ | Full support ✅ |
| ChromaDB | Requires C++ tools | Requires build-essential | Requires Xcode tools |

---

## 🚀 Quick Start Guide

### Windows
```powershell
# Setup
.\setup.ps1
copy .env.example .env
notepad .env  # Add API key

# Activate and run
.\venv\Scripts\Activate.ps1
python simple_cli.py "Your prompt" --no-emoji
```

### Linux
```bash
# Setup
chmod +x setup.sh
./setup.sh
cp .env.example .env
nano .env  # Add API key

# Activate and run
source venv/bin/activate
python simple_cli.py "Your prompt" --no-emoji
```

### macOS
```bash
# Setup
chmod +x setup.sh
./setup.sh
cp .env.example .env
nano .env  # Add API key

# Activate and run
source venv/bin/activate
python simple_cli.py "Your prompt" --no-emoji
```

---

## 📦 Platform-Specific Dependencies

### Windows
**C++ Build Tools** (for ChromaDB):
- Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Select: "Desktop development with C++"

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y python3-dev python3-venv build-essential
```

### Linux (Fedora/RHEL/CentOS)
```bash
sudo yum install -y python3-devel gcc gcc-c++
```

### macOS
```bash
# Install Xcode Command Line Tools
xcode-select --install
```

---

## 🛠️ Key Differences Handled

### 1. **Activation Commands**
- Windows: `.\venv\Scripts\Activate.ps1`
- Linux/macOS: `source venv/bin/activate`

### 2. **Python Executable Name**
- Windows: Usually `python`
- Linux/macOS: Often `python3` (handled with detection in setup.sh)

### 3. **File Operations**
- Copy: `copy` (Windows) vs `cp` (Linux/macOS)
- Edit: `notepad` (Windows) vs `nano`/`vim` (Linux/macOS)
- View: `Get-Content` (PowerShell) vs `cat` (Bash)

### 4. **Path Separators**
- Windows: `tools\inspect_library.py`
- Linux/macOS: `tools/inspect_library.py`
- **Note**: Python handles both on all platforms!

### 5. **Executable Permissions**
- Linux/macOS: Must `chmod +x` before running `.sh` files
- Windows: Not required for `.ps1` files (but may need execution policy)

---

## ✅ Testing Checklist

### Windows Testing
- [ ] `.\setup.ps1` runs without errors
- [ ] Virtual environment activates
- [ ] `python simple_cli.py "test" --mock` works
- [ ] Tools run: `python tools\inspect_library.py`

### Linux Testing
- [ ] `./setup.sh` runs without errors
- [ ] Virtual environment activates
- [ ] `python simple_cli.py "test" --mock` works
- [ ] Tools run: `python tools/inspect_library.py`

### macOS Testing
- [ ] `./setup.sh` runs without errors
- [ ] Virtual environment activates
- [ ] `python simple_cli.py "test" --mock` works
- [ ] Tools run: `python tools/inspect_library.py`

---

## 📝 Documentation Files

| File | Purpose | Platform |
|------|---------|----------|
| `setup.ps1` | Setup script | Windows |
| `setup.sh` | Setup script | Linux/macOS |
| `GETTING_STARTED.md` | Complete beginner guide | All platforms |
| `DOCUMENTATION.md` | Full system documentation | All platforms |
| `README.md` | Project overview | All platforms |

---

## 🎯 Common Commands Reference

### Setup (One-time)
| Task | Windows | Linux/macOS |
|------|---------|-------------|
| Run setup | `.\setup.ps1` | `./setup.sh` |
| Copy env file | `copy .env.example .env` | `cp .env.example .env` |
| Edit env file | `notepad .env` | `nano .env` |

### Daily Use
| Task | Windows | Linux/macOS |
|------|---------|-------------|
| Activate venv | `.\venv\Scripts\Activate.ps1` | `source venv/bin/activate` |
| Run optimizer | `python simple_cli.py "prompt"` | `python simple_cli.py "prompt"` |
| View library | `python tools\inspect_library.py` | `python tools/inspect_library.py` |
| View database | `python tools\show_all_db.py` | `python tools/show_all_db.py` |
| Deactivate | `deactivate` | `deactivate` |

---

## 🔧 Troubleshooting by Platform

### Windows-Specific Issues
1. **PowerShell execution policy**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

2. **C++ build tools missing**
   - Install Visual Studio Build Tools
   - Or fallback to SQL-only mode (automatic)

### Linux-Specific Issues
1. **Permission denied on setup.sh**
   ```bash
   chmod +x setup.sh
   ```

2. **python3-dev missing**
   ```bash
   sudo apt install python3-dev build-essential
   ```

### macOS-Specific Issues
1. **Xcode tools not installed**
   ```bash
   xcode-select --install
   ```

2. **SSL certificate errors**
   ```bash
   /Applications/Python\ 3.11/Install\ Certificates.command
   ```

---

## 🎉 Success!

The project now supports:
- ✅ Windows (PowerShell)
- ✅ Linux (Bash)
- ✅ macOS (Bash)

All documentation has been updated with platform-specific instructions and examples.

**Users can now follow `GETTING_STARTED.md` on any platform!**
