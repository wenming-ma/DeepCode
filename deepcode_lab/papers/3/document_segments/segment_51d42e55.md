# Related Work

**Content Type:** algorithm
**Keywords:** development, alphaevolve, modular, open, workflows, frontiers, failing, paradigms, synthesis, designer

Related Work
5.1 General Coding Agents
The field of software engineering is being rapidly transformed by agentic systems that have evolved
from passive code assistants into autonomous entities capable of planning, executing multi-step
tasks, and self-correction [ 4,2]. Research has explored several key architectures for these agents.
One prominent trend involves multi-agent frameworks that emulate human development teams.
This includes systems like ChatDev [ 12], MetaGPT [ 13], and CodePoRi [ 14], which simulate entire
software company organizational structures to manage development tasks from scratch. For repo-level
code generation, CodeS [ 15] proposed to decompose repository generation into specialized agents
for structure planning and content filling. AgentCoder [ 16] employs atest-driven refinement loop
involving programmer, test designer, and test executor agents, while MapCoder [ 17] mirrors human
program synthesis with four agents handling example retrieval, planning, generation, and debugging.
A second major trend focuses on enhancing agents with specialized tools and interfaces. For instance,
CodeAgent [ 18] integrates five domain-specific tools to support repository-level analysis, while SWE-
agent [ 19] introduces a high-level Agent-Computer Interface (ACI) to enable robust agent interaction
with file systems and development environments. In addition, ToolGen [ 20] proposes representing
each tool as a unique token and directly integrating tool-specific knowledge into the parameters of the
LLM, thereby enabling a paradigm shift toward seamless unification of tool invocation and natural
language generation.
Recent advancements in academic research are increasingly being translated into practical, produc-
tized tools. Commercial code agents emerging from this trend can be broadly categorized into two
distinct paradigms: (1) AI-native integrated development environments (IDEs) such as Cursor [ 9]
and Trae [ 21] that embed AI capabilities directly into the editor interface, and (2) terminal-based
or extension-based agents including Claude Code [ 10], Gemini CLI [ 22], Github Copilot [ 23], and
Cline [ 24] that operate through command-line interfaces or editor extensions. These coding agents
leverage a holistic understanding of the codebase to perform complex tasks such as multi-file refac-
toring and autonomous edits. They support flexible, composable workflows and integrate seamlessly
into diverse development pipelines. Commercial deployments indicate significant improvements in
both function implementation and overall programming productivity. Despite their effectiveness,
these agents suffer from context window limitations that impair their ability to process lengthy
technical documents such as academic papers, and struggle to maintain coherence and correctness
when synthesizing repository-level codebases.
5.2 Scientific Coding Agents
In contrast to general-purpose coding agents, this class of agents targets more complex code generation
scenarios, including the implementation and reproduction of entire codebases from high-level ideas
and academic papers. For example, Paper2Code [ 8] addresses the research reproducibility crisis by
transforming machine learning papers into executable repositories. Its code generation framework
follows a structured three-stage process that includes system architecture design, implementation
detail extraction, and modular code generation. CodeScientist [ 25] generates experimental code
from literature, employing an iterative generate-execute-reflect cycle to write, run, and debug Python
experiments. In addition, AlphaEvolve [ 26] utilize code generation for algorithmic discovery, using
an LLM as an evolutionary mutator to propose variations to entire codebases, which are then
rigorously evaluated. Besides, the automation code in AI Scientist [ 27] and AI-Researcher [ 6]
enables agents to iteratively plan and execute experiments, handle errors, and refine future runs
based on results. AI Scientist focuses on experimental automation, maintaining execution history
and generating plots and notes to support scientific write-ups. AI-Researcher extends this with a
multi-stage refinement framework, where a code agent implements modular solutions and an advisor
agent provides structured feedback for iterative validation, revision, and scaling. These agents
have advanced the pace of scientific research, yet achieving higher generation efficiency without
compromising code quality remains an open challenge.
14

## Page 15

6 Discussion: Challenges and Future Directions
While DeepCode demonstrates the efficacy of principled information-flow management in high-
fidelity repository synthesis, the transition from episodic coding tasks to autonomous, cost-effective,
and self-evolving engineering remains fraught with challenges. We identify three critical frontiers
that define the future trajectory of agentic software engineering.
(1) Agentic Capability and Computational Efficiency.SOTA performance in agentic coding
currently relies on massive, proprietary LLMs (e.g. GPT-5, Claude 4.5), which incur prohibitive
deployment costs and high latency. Conversely, smaller, open-weight models offer efficiency but
lack the complex reasoning capabilities required for autonomous decision-making in open-ended
engineering tasks. Bridging this gap presents a dichotomy of challenges.(i) Fine-tuning limits:
Enhancing small models via supervised fine-tuning (SFT) is constrained by a data bottleneck—while
raw code is abundant, high-quality agentic trajectories are scarce and expensive to curate.(ii)
Knowledge injection limits:Merely augmenting small models with external knowledge is often
insufficient; retrieved contexts may lack direct relevance to the specific coding task, and small models
struggle to integrate complex inputs without suffering from attention dilution.
We envision a shift toward hybrid agentic architectures that synergize models of varying scales, em-
ploying large models for high-level reasoning and efficient small models for routine implementation.
Besides, distilling knowledge from large models helps reduce the data bottleneck.
(2) From Episodic to Evolving Agents.Current coding agents typically operate in an episodic
manner: they reset after each project, failing to carry over experience or tacit knowledge to subsequent
tasks. Enabling agents to self-evolve and accumulate expertise mirrors human professional growth
but faces significant hurdles.(i) Reinforcement Learning constraints:While RL-based optimization
theoretically allows agents to learn from feedback, it requires well-defined reward functions, which
are difficult to formulate for complex, multi-objective software engineering tasks. Moreover, this