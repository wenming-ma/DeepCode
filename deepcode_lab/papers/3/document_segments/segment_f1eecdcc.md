# Abstract

**Content Type:** introduction
**Keywords:** fidelity, given, bottlenecks, challenge, lizongwei, code, large, budgets, figure, existing

Abstract
Recent advances in large language models (LLMs) have given rise to powerful
coding agents, making it possible for code assistants to evolve into code engineers.
However, existing methods still face significant challenges in achieving high-fidelity
document-to-codebase synthesis—such as scientific papers to code—primarily due
to a fundamental conflict between information overload and the context bottlenecks
of LLMs. In this work, we introduce DeepCode, a fully autonomous framework
that fundamentally addresses this challenge through principled information-flow
management. By treating repository synthesis as a channel optimization problem,
DeepCode seamlessly orchestrates four information operations to maximize task-
relevant signals under finite context budgets: source compression via blueprint
distillation, structured indexing using stateful code memory, conditional knowl-
edge injection via retrieval-augmented generation, and closed-loop error correction.
Extensive evaluations on the PaperBench benchmark demonstrate that DeepCode
achieves state-of-the-art performance, decisively outperforming leading commer-
cial agents such as Cursor and Claude Code, and crucially, surpassing PhD-level
human experts from top institutes on key reproduction metrics. By systematically
transforming paper specifications into production-grade implementations compara-
ble to human expert quality, this work establishes new foundations for autonomous
scientific reproduction that can accelerate research evaluation and discovery.
① Human Expert (Top ML PhD)
100
75
50
25
0
72.4%75.9%
Human Expert
DeepCode② Commercial Code Agents
100
75
50
25
0
40.0%58.7%58.4%84.8%
Codex
Claude Code
Cursor
DeepCode③ Scientific Code Agent
100
75
50
25
0
51.1%73.5%
Paper Coder
DeepCode④ LLM-Based Agents
100
75
50
25
0
5.0%7.7%9.8%16.4%35.4%43.3%73.5%
Gemini 2-flash
GPT-4o
DeepSeek R1
o3-mini
Claude 3.5o1
DeepCode08/12/2025, 22:43 New_UI.html
ﬁle:///Users/lizongwei/Desktop/new_draw/New_UI.html 1/1
Figure 1: DeepCode main results.
1