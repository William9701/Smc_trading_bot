# setup_windows.bat
# ==================
# Windows Setup Script

@echo off
echo ===========================================
echo SMC Trading Bot - Windows Setup
echo ===========================================

echo Checking Python version...
python --version
echo.

echo Creating virtual environment...
python -m venv smc_trading_bot_env
echo Virtual environment created!
echo.

echo Activating virtual environment...
call smc_trading_bot_env\Scripts\activate
echo.

echo Upgrading pip...
python -m pip install --upgrade pip
echo.

echo Installing core dependencies...
pip install MetaTrader5==5.0.45 pandas==2.1.4 numpy==1.24.3 pytz==2023.3
echo Core dependencies installed!
echo.

echo Installing web framework...
pip install fastapi==0.104.1 uvicorn[standard]==0.24.0 pydantic==2.5.0 pydantic-settings==2.1.0
echo Web framework installed!
echo.

echo Installing development tools...
pip install pytest==7.4.3 loguru==0.7.2 python-dotenv==1.0.0 psutil==5.9.6
echo Development tools installed!
echo.

echo Installing visualization tools...
pip install plotly==5.17.0 dash==2.14.2 matplotlib==3.8.0
echo Visualization tools installed!
echo.

echo Installing ML libraries (this may take a few minutes)...
pip install scikit-learn==1.3.0 tensorflow==2.15.0 scipy==1.11.4
echo ML libraries installed!
echo.

echo Creating project directories...
mkdir data_service
mkdir market_structure
mkdir pattern_recognition
mkdir liquidity_service
mkdir premium_discount
mkdir config
mkdir utils
mkdir tests
mkdir logs
mkdir reports
echo Project structure created!
echo.

echo Creating .env template...
echo # SMC Trading Bot Configuration > .env.example
echo MT5_LOGIN=your_mt5_login >> .env.example
echo MT5_PASSWORD=your_mt5_password >> .env.example
echo MT5_SERVER=your_mt5_server >> .env.example
echo DEBUG=False >> .env.example
echo LOG_LEVEL=INFO >> .env.example
echo.

echo ===========================================
echo Setup Complete! 
echo ===========================================
echo.
echo Next steps:
echo 1. Copy .env.example to .env
echo 2. Edit .env with your MT5 credentials
echo 3. Run: python main.py
echo.
echo To activate environment later:
echo call smc_trading_bot_env\Scripts\activate
echo.
pause

# setup_mac_linux.sh
# ==================
# Mac/Linux Setup Script

#!/bin/bash

echo "==========================================="
echo "SMC Trading Bot - Mac/Linux Setup"
echo "==========================================="

# Check Python version
echo "Checking Python version..."
python3 --version
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv smc_trading_bot_env
echo "Virtual environment created!"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source smc_trading_bot_env/bin/activate
echo ""

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip
echo ""

# Install core dependencies
echo "Installing core dependencies..."
pip install MetaTrader5==5.0.45 pandas==2.1.4 numpy==1.24.3 pytz==2023.3
echo "Core dependencies installed!"
echo ""

# Install web framework
echo "Installing web framework..."
pip install fastapi==0.104.1 uvicorn[standard]==0.24.0 pydantic==2.5.0 pydantic-settings==2.1.0
echo "Web framework installed!"
echo ""

# Install development tools
echo "Installing development tools..."
pip install pytest==7.4.3 loguru==0.7.2 python-dotenv==1.0.0 psutil==5.9.6
echo "Development tools installed!"
echo ""

# Install visualization tools
echo "Installing visualization tools..."
pip install plotly==5.17.0 dash==2.14.2 matplotlib==3.8.0
echo "Visualization tools installed!"
echo ""

# Install ML libraries
echo "Installing ML libraries (this may take a few minutes)..."
pip install scikit-learn==1.3.0 tensorflow==2.15.0 scipy==1.11.4
echo "ML libraries installed!"
echo ""

# Create project directories
echo "Creating project directories..."
mkdir -p data_service
mkdir -p market_structure
mkdir -p pattern_recognition
mkdir -p liquidity_service
mkdir -p premium_discount
mkdir -p config
mkdir -p utils
mkdir -p tests
mkdir -p logs
mkdir -p reports
echo "Project structure created!"
echo ""

# Create .env template
echo "Creating .env template..."
cat > .env.example << EOF
# SMC Trading Bot Configuration
MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_mt5_server
DEBUG=False
LOG_LEVEL=INFO
EOF
echo ""

echo "==========================================="
echo "Setup Complete!"
echo "==========================================="
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env"
echo "2. Edit .env with your MT5 credentials"
echo "3. Run: python main.py"
echo ""
echo "To activate environment later:"
echo "source smc_trading_bot_env/bin/activate"
echo ""

# quick_install.py
# ================
# Python Quick Install Script (Works on all platforms)

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 SMC Trading Bot - Quick Setup")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major != 3 or python_version.minor < 9:
        print("❌ Python 3.9+ required. Please upgrade Python.")
        return False
    
    if python_version.minor > 10:
        print("⚠️  Warning: Python 3.11+ may have MetaTrader5 compatibility issues.")
        print("   Recommended: Python 3.9.18 or 3.10.12")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return False
    
    # Create virtual environment
    if not run_command("python -m venv smc_trading_bot_env", "Creating virtual environment"):
        return False
    
    # Determine activation command
    system = platform.system()
    if system == "Windows":
        activate_cmd = "smc_trading_bot_env\\Scripts\\activate"
        pip_cmd = "smc_trading_bot_env\\Scripts\\pip"
    else:
        activate_cmd = "source smc_trading_bot_env/bin/activate"
        pip_cmd = "smc_trading_bot_env/bin/pip"
    
    # Upgrade pip
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install dependencies in groups
    core_deps = "MetaTrader5==5.0.45 pandas==2.1.4 numpy==1.24.3 pytz==2023.3"
    web_deps = "fastapi==0.104.1 uvicorn[standard]==0.24.0 pydantic==2.5.0 pydantic-settings==2.1.0"
    dev_deps = "pytest==7.4.3 loguru==0.7.2 python-dotenv==1.0.0 psutil==5.9.6"
    viz_deps = "plotly==5.17.0 dash==2.14.2 matplotlib==3.8.0"
    
    install_groups = [
        (core_deps, "Installing core dependencies"),
        (web_deps, "Installing web framework"), 
        (dev_deps, "Installing development tools"),
        (viz_deps, "Installing visualization tools")
    ]
    
    for deps, description in install_groups:
        if not run_command(f"{pip_cmd} install {deps}", description):
            print(f"⚠️  {description} failed, but continuing...")
    
    # ML libraries (optional, may take time)
    print("\n🤖 Installing ML libraries (optional, may take 5-10 minutes)...")
    response = input("Install ML libraries now? (Y/n): ")
    
    if response.lower() != 'n':
        ml_deps = "scikit-learn==1.3.0 tensorflow==2.15.0 scipy==1.11.4"
        run_command(f"{pip_cmd} install {ml_deps}", "Installing ML libraries")
    
    # Create project structure
    directories = [
        "data_service", "market_structure", "pattern_recognition",
        "liquidity_service", "premium_discount", "config", "utils",
        "tests", "logs", "reports"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Create .env template
    env_template = """# SMC Trading Bot Configuration
MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_mt5_server
DEBUG=False
LOG_LEVEL=INFO
"""
    
    with open(".env.example", "w") as f:
        f.write(env_template)
    
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Copy .env.example to .env")
    print("2. Edit .env with your MT5 credentials")
    print("3. Activate environment:")
    if system == "Windows":
        print("   smc_trading_bot_env\\Scripts\\activate")
    else:
        print("   source smc_trading_bot_env/bin/activate")
    print("4. Run: python main.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)