@echo off
REM Download script for DeepCode paper reference repositories
REM Run this script from the code_base directory

echo ============================================
echo Downloading GitHub repositories for DeepCode paper references
echo ============================================

REM Create subdirectories for each repository
mkdir MetaGPT 2>nul
mkdir SWE-agent 2>nul
mkdir ChatDev 2>nul
mkdir AI-Scientist 2>nul
mkdir Cline 2>nul

echo.
echo [1/5] Cloning MetaGPT (60.8k stars) - Multi-agent software development framework
git clone https://github.com/geekan/MetaGPT.git MetaGPT
echo.

echo [2/5] Cloning SWE-agent (18k stars) - Agent-Computer Interface for software engineering
git clone https://github.com/SWE-agent/SWE-agent.git SWE-agent
echo.

echo [3/5] Cloning ChatDev (27.9k stars) - Communicative agents for software development
git clone https://github.com/OpenBMB/ChatDev.git ChatDev
echo.

echo [4/5] Cloning AI-Scientist (11.8k stars) - Automated scientific discovery
git clone https://github.com/SakanaAI/AI-Scientist.git AI-Scientist
echo.

echo [5/5] Cloning Cline (56.2k stars) - Autonomous coding agent for VS Code
git clone https://github.com/cline/cline.git Cline
echo.

echo ============================================
echo Download complete!
echo ============================================
echo.
echo Repository Summary:
echo - MetaGPT: Multi-agent framework with role-based collaboration
echo - SWE-agent: Agent-Computer Interface for automated software engineering
echo - ChatDev: Virtual software company simulation
echo - AI-Scientist: Fully automated scientific discovery
echo - Cline: IDE-integrated autonomous coding with MCP support
echo.
pause
