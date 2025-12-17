# PowerShell script for downloading DeepCode paper reference repositories
# Run this script from the code_base directory

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Downloading GitHub repositories for DeepCode paper references" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$repos = @(
    @{
        Name = "MetaGPT"
        Url = "https://github.com/geekan/MetaGPT.git"
        Stars = "60.8k"
        Description = "Multi-agent software development framework with role-based collaboration"
    },
    @{
        Name = "SWE-agent"
        Url = "https://github.com/SWE-agent/SWE-agent.git"
        Stars = "18k"
        Description = "Agent-Computer Interface for automated software engineering"
    },
    @{
        Name = "ChatDev"
        Url = "https://github.com/OpenBMB/ChatDev.git"
        Stars = "27.9k"
        Description = "Communicative agents for software development"
    },
    @{
        Name = "AI-Scientist"
        Url = "https://github.com/SakanaAI/AI-Scientist.git"
        Stars = "11.8k"
        Description = "Fully automated scientific discovery"
    },
    @{
        Name = "Cline"
        Url = "https://github.com/cline/cline.git"
        Stars = "56.2k"
        Description = "Autonomous coding agent for VS Code with MCP support"
    }
)

$total = $repos.Count
$current = 0

foreach ($repo in $repos) {
    $current++
    Write-Host ""
    Write-Host "[$current/$total] Cloning $($repo.Name) ($($repo.Stars) stars)" -ForegroundColor Green
    Write-Host "        $($repo.Description)" -ForegroundColor Gray
    Write-Host ""
    
    if (Test-Path $repo.Name) {
        Write-Host "        Directory exists, pulling latest changes..." -ForegroundColor Yellow
        Push-Location $repo.Name
        git pull
        Pop-Location
    } else {
        git clone $repo.Url $repo.Name
    }
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Download complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Repository Summary:" -ForegroundColor Yellow
Write-Host "- MetaGPT: Multi-agent framework with role-based collaboration (SOPs)" -ForegroundColor White
Write-Host "- SWE-agent: Agent-Computer Interface for automated software engineering" -ForegroundColor White
Write-Host "- ChatDev: Virtual software company simulation with multiple agent roles" -ForegroundColor White
Write-Host "- AI-Scientist: Fully automated scientific discovery and paper generation" -ForegroundColor White
Write-Host "- Cline: IDE-integrated autonomous coding with MCP support" -ForegroundColor White
Write-Host ""
Write-Host "These repositories provide implementation patterns for:" -ForegroundColor Cyan
Write-Host "  - Multi-agent architecture (DeepCode's Concept/Algorithm/Code Planning Agents)" -ForegroundColor Gray
Write-Host "  - Agent-Computer Interface (Sandbox execution, file operations)" -ForegroundColor Gray
Write-Host "  - Role-based collaboration (Blueprint generation)" -ForegroundColor Gray
Write-Host "  - Automated verification (Iterative refinement loops)" -ForegroundColor Gray
Write-Host "  - MCP integration (Tool calling, external services)" -ForegroundColor Gray
Write-Host ""
