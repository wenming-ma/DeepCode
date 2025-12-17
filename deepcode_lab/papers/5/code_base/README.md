# DeepCode Paper - Reference Code Repositories

This directory contains the code repositories from the key references cited in the DeepCode paper that have high-quality GitHub implementations.

## Quick Start

### Option 1: Using PowerShell (Recommended)
```powershell
cd C:\Users\wenming\source\repos\DeepCode\deepcode_lab\papers\5\code_base
.\download_repos.ps1
```

### Option 2: Using Command Prompt
```cmd
cd C:\Users\wenming\source\repos\DeepCode\deepcode_lab\papers\5\code_base
download_repos.bat
```

### Option 3: Manual Clone
```bash
git clone https://github.com/geekan/MetaGPT.git
git clone https://github.com/SWE-agent/SWE-agent.git
git clone https://github.com/OpenBMB/ChatDev.git
git clone https://github.com/SakanaAI/AI-Scientist.git
git clone https://github.com/cline/cline.git
```

---

## Repository Overview

### 1. MetaGPT (⭐ 60.8k)
**URL:** https://github.com/geekan/MetaGPT

**Relevance to DeepCode:** Multi-agent software development framework with Standard Operating Procedures (SOPs) for agent collaboration.

**Key Features:**
- Multi-agent software company simulation
- Role-based agent architecture (Product Manager, Architect, Engineer)
- Natural language to code generation
- Data Interpreter for automated coding
- AFlow for agentic workflow generation

**Implementation Value for DeepCode:**
- Provides patterns for DeepCode's multi-agent architecture (Concept Agent, Algorithm Agent)
- SOPs for orchestrating agent collaboration
- Role specialization patterns for blueprint generation

---

### 2. SWE-agent (⭐ 18k)
**URL:** https://github.com/SWE-agent/SWE-agent

**Relevance to DeepCode:** Agent-Computer Interface (ACI) design for robust agent-environment interaction.

**Key Features:**
- Autonomous GitHub issue fixing
- Configurable via single YAML file
- Mini-SWE-Agent (65% on SWE-bench verified in 100 lines)
- Offensive cybersecurity capabilities (EnIGMA)

**Implementation Value for DeepCode:**
- Critical patterns for sandbox execution
- File system and development environment interaction
- LSP-based code modification features
- Automated verification approaches

---

### 3. ChatDev (⭐ 27.9k)
**URL:** https://github.com/OpenBMB/ChatDev

**Relevance to DeepCode:** Virtual software company simulation with organizational structure for managing development tasks.

**Key Features:**
- Multi-agent organizational structure (CEO, CTO, Programmer, Reviewer, Tester)
- Functional seminars (design, coding, testing, documenting)
- Multi-Agent Collaboration Networks (MacNet)
- Customizable company configurations

**Implementation Value for DeepCode:**
- Role specialization and inter-agent communication patterns
- Task decomposition through organizational structure
- Verification through reviewer and tester agents

---

### 4. AI-Scientist (⭐ 11.8k)
**URL:** https://github.com/SakanaAI/AI-Scientist

**Relevance to DeepCode:** Automation code for iterative experiment planning, execution, and refinement.

**Key Features:**
- Fully automated paper generation
- Multiple research templates (NanoGPT, 2D Diffusion, Grokking)
- LLM-based experiment iteration
- Automatic idea generation and testing

**Implementation Value for DeepCode:**
- Automated verification and iterative refinement loops
- Error handling and correction patterns
- Generate-execute-reflect cycle implementation

---

### 5. Cline (⭐ 56.2k)
**URL:** https://github.com/cline/cline

**Relevance to DeepCode:** Terminal-based/extension-based autonomous coding agent with MCP integration.

**Key Features:**
- Autonomous file creation and editing
- Terminal command execution with human-in-the-loop approval
- Browser automation for web testing
- Model Context Protocol (MCP) support
- Multi-model support (Claude, GPT-4o, etc.)

**Implementation Value for DeepCode:**
- File system operations patterns
- Command execution with approval workflow
- MCP tool integration
- Browser-based testing for web UI generation

---

## Mapping to DeepCode Architecture

| DeepCode Component | Relevant Repositories |
|-------------------|----------------------|
| **Concept Agent** | MetaGPT (role-based agents), ChatDev (organizational structure) |
| **Algorithm Agent** | MetaGPT (SOPs), AI-Scientist (experiment design) |
| **Code Planning Agent** | MetaGPT (architect role), ChatDev (design seminar) |
| **CodeRAG** | SWE-agent (file system interaction), Cline (context management) |
| **Blueprint Generation** | MetaGPT (project structure), ChatDev (task decomposition) |
| **Sandbox Execution** | SWE-agent (ACI), Cline (terminal execution) |
| **Automated Verification** | AI-Scientist (experiment iteration), SWE-agent (testing) |
| **MCP Integration** | Cline (MCP support), MetaGPT (tool integration) |

---

## Citation Information

### MetaGPT
```bibtex
@article{hong2024metagpt,
  title={MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework},
  author={Hong, Sirui and Zhuge, Mingchen and Chen, Jiaqi and others},
  journal={arXiv preprint arXiv:2308.00352},
  year={2024}
}
```

### SWE-agent
```bibtex
@inproceedings{yang2024sweagent,
  title={SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering},
  author={Yang, John and Jimenez, Carlos E and Wettig, Alexander and others},
  booktitle={NeurIPS},
  year={2024}
}
```

### ChatDev
```bibtex
@article{qian2024chatdev,
  title={ChatDev: Communicative Agents for Software Development},
  author={Qian, Chen and Liu, Wei and Liu, Hongzhang and others},
  journal={arXiv preprint arXiv:2307.07924},
  year={2024}
}
```

### AI-Scientist
```bibtex
@article{lu2024aiscientist,
  title={The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery},
  author={Lu, Chris and Lu, Cong and Lange, Robert Tjarko and others},
  journal={arXiv preprint arXiv:2408.06292},
  year={2024}
}
```

### Cline
```bibtex
@misc{rizwan2024cline,
  title={Cline: Autonomous Coding Agent for VS Code},
  author={Rizwan, Saoud and others},
  howpublished={\url{https://github.com/cline/cline}},
  year={2024}
}
```

---

## Additional Resources

For more repositories mentioned in the DeepCode paper (but not included in the top 5):

- **Auto-GPT**: https://github.com/Significant-Gravitas/Auto-GPT
- **AgentCoder**: https://github.com/huangd1999/AgentCoder
- **CodeGeeX**: https://github.com/THUDM/CodeGeeX
- **OpenCodeInterpreter**: https://github.com/OpenCodeInterpreter/OpenCodeInterpreter
- **AlphaCodium**: https://github.com/Codium-ai/AlphaCodium
