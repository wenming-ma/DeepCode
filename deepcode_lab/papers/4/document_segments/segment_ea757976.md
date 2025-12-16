# approach

**Content Type:** methodology
**Keywords:** ability, best, mert, alphaevolve, linear, paperbench, issues, agentic, dynamic, computer

approach is inapplicable to closed-source LLMs where parameter updates are impossible.(ii)
Memory scalability issues:The alternative approach—stacking historical experiences into a long-
term memory—introduces severe noise. Simply accumulating raw interaction logs leads to context
bloat, where retrieving relevant past experiences becomes a “needle in a haystack” problem.
Beyond relying on extensive manual annotation and training, a scalable solution involves automating
the abstraction of past experiences. Future agents can implement post-task reflection to condense
execution traces into reusable skills or heuristics. Storing these refined insights allows agents to
retrieve corresponding high-level guidance, enabling self-evolution while avoiding context explosion.
(3) Dynamic Planning and Adaptability.Most existing frameworks utilize a linear Plan-then-Code
workflow, assuming that all constraints are knowable a priori. In real-world engineering, specifications
often evolve, and critical implementation constraints are frequently discovered only during the coding
process. Separation between planning and execution leads to fragility: if the initial blueprint is flawed,
the coding agent is often constrained by a stale plan, leading to suboptimal workarounds or failure.
Future researches advance toward dynamic, bidirectional planning frameworks. Agents are able
to adapt their initial blueprints when encountering unforeseen constraints during implementation.
Establishing a feedback mechanism where execution insights directly inform and update the high-level
plan is crucial for handling the complex realities of large-scale software development.
7 Conclusion
In this work, we presented DeepCode, an autonomous framework that advances the frontier of agentic
code engineering by reimagining document-to-repository synthesis as a challenge ofinformation-flow
management. Addressing the fundamental conflict between information overload and finite context
bottlenecks, we demonstrated that treating synthesis as a channel optimization problem—solved
through the orchestration of blueprint distillation, stateful memory, conditional knowledge injection,
and closed-loop verification—effectively maximizes the signal-to-noise ratio for long-horizon tasks.
Empirical evaluations on PaperBench confirm that DeepCode establishes a new SOTA, decisively
outperforming leading commercial agents and surpassing PhD-level human experts in reproduction
accuracy. These findings validate that hierarchical information orchestration, rather than indiscrimi-
nate context scaling, provides the decisive path toward robust autonomous systems, laying a critical
foundation for the future of automated scientific discovery and rigorous research reproduction.
15

## Page 16

References
[1]Juyong Jiang, Fan Wang, Jiasi Shen, Sungju Kim, and Sunghun Kim. A survey on large
language models for code generation.arXiv preprint arXiv:2406.00515, 2024.
[2]Yuyao Ge, Lingrui Mei, Zenghao Duan, Tianhao Li, Yujia Zheng, Yiwei Wang, Lexin Wang,
Jiayu Yao, Tianyu Liu, Yujun Cai, Baolong Bi, Fangda Guo, Jiafeng Guo, Shenghua Liu,
and Xueqi Cheng. A survey of vibe coding with large language models, 2025. URL https:
//arxiv.org/abs/2510.12399.
[3]Sida Peng, Eirini Kalliamvakou, Peter Cihon, and Mert Demirer. The impact of ai on developer
productivity: Evidence from github copilot.arXiv preprint arXiv:2302.06590, 2023.
[4]Yihong Dong, Xue Jiang, Jiaru Qian, Tian Wang, Kechi Zhang, Zhi Jin, and Ge Li. A survey on
code generation with llm-based agents, 2025. URL https://arxiv.org/abs/2508.00083 .
[5]Huanting Wang, Jingzhi Gong, Huawei Zhang, Jie Xu, and Zheng Wang. Ai agentic program-
ming: A survey of techniques, challenges, and opportunities.arXiv preprint arXiv:2508.11126,
2025.
[6]Jiabin Tang, Lianghao Xia, Zhonghang Li, and Chao Huang. AI-Researcher: Autonomous
Scientific Innovation. InNeurIPS, 2025.
[7]Giulio Starace, Oliver Jaffe, Dane Sherburn, James Aung, Jun Shern Chan, Leon Maksin,
Rachel Dias, Evan Mays, Benjamin Kinsella, Wyatt Thompson, Johannes Heidecke, Amelia
Glaese, and Tejal Patwardhan. Paperbench: Evaluating ai’s ability to replicate ai research, 2025.
URLhttps://arxiv.org/abs/2504.01848.
[8]Minju Seo, Jinheon Baek, Seongyun Lee, and Sung Ju Hwang. Paper2code: Automating code
generation from scientific papers in machine learning, 2025. URL https://arxiv.org/abs/
2504.17192.
[9] Anysphere. Cursor: The best way to code with ai.https://cursor.com, 2025.
[10] Anthropic. Claude code: Agentic coding tool for your terminal. https://docs.claude.com/
en/docs/claude-code/overview, 2025.
[11] OpenAI. Codex cli: Pair with codex in your terminal. https://developers.openai.com/
codex/cli, 2025.
[12] Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize
Chen, Yusheng Su, Xin Cong, Juyuan Xu, Dahai Li, Zhiyuan Liu, and Maosong Sun. Chatdev:
Communicative agents for software development, 2024. URL https://arxiv.org/abs/
2307.07924.
[13] Sirui Hong, Mingchen Zhuge, Jiaqi Chen, Xiawu Zheng, Yuheng Cheng, Ceyao Zhang, Jinlin
Wang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng
Xiao, Chenglin Wu, and Jürgen Schmidhuber. Metagpt: Meta programming for a multi-agent
collaborative framework, 2024. URLhttps://arxiv.org/abs/2308.00352.
[14] Zeeshan Rasheed, Malik Abdul Sami, Kai-Kristian Kemell, Muhammad Waseem, Mika Saari,
Kari Systä, and Pekka Abrahamsson. Codepori: Large-scale system for autonomous software
development using multi-agent technology, 2024. URL https://arxiv.org/abs/2402.
01411.
[15] Daoguang Zan, Ailun Yu, Wei Liu, Dong Chen, Bo Shen, Yafen Yao, Wei Li, Xiaolin Chen,
Yongshun Gong, Bei Guan, et al. Codes: Natural language to code repository via multi-layer
sketch.ACM Transactions on Software Engineering and Methodology, 2024.
[16] Dong Huang, Jie M. Zhang, Michael Luck, Qingwen Bu, Yuhao Qing, and Heming Cui.
Agentcoder: Multi-agent-based code generation with iterative testing and optimisation, 2024.
URLhttps://arxiv.org/abs/2312.13010.
[17] Md. Ashraful Islam, Mohammed Eunus Ali, and Md Rizwan Parvez. MapCoder: Multi-agent
code generation for competitive problem solving. InACL, pages 4912–4944, 2024.
16

## Page 17

[18] Kechi Zhang, Jia Li, Ge Li, Xianjie Shi, and Zhi Jin. CodeAgent: Enhancing code generation
with tool-integrated agent systems for real-world repo-level coding challenges. InACL, pages
13643–13658, 2024.
[19] John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik
Narasimhan, and Ofir Press. Swe-agent: agent-computer interfaces enable automated software
engineering. InNeurIPS, pages 50528–50652, 2025.
[20] Renxi Wang, Xudong Han, Lei Ji, Shu Wang, Timothy Baldwin, and Haonan Li. Toolgen:
Unified tool retrieval and calling via generation. InThe Thirteenth International Conference on
Learning Representations, 2025. URL https://openreview.net/forum?id=XLMAMmowdY .
[21] ByteDance. Trae: The real ai engineer.https://www.trae.ai, 2025.
[22] Google. Gemini cli: An open-source ai agent that brings the power of gemini directly into your
terminal.https://github.com/google-gemini/gemini-cli, 2025.
[23] GitHub and OpenAI. Github copilot: Your ai pair programmer. https://github.com/
features/copilot, 2025.
[24] Saoud Rizwan and others. Cline: Autonomous coding agent for vs code. https://github.
com/cline/cline, 2024.
[25] Peter Jansen, Oyvind Tafjord, Marissa Radensky, Pao Siangliulue, Tom Hope, Bhavana Dalvi
Mishra, Bodhisattwa Prasad Majumder, Daniel S. Weld, and Peter Clark. Codescientist: End-
to-end semi-automated scientific discovery with code-based experimentation, 2025. URL
https://arxiv.org/abs/2503.22708.
[26] Alexander Novikov, Ngân V ˜u, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, et al.
Alphaevolve: A coding agent for scientific and