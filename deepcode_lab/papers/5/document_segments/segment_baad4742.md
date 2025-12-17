# Introduction

**Content Type:** introduction
**Keywords:** manipulation, full, indicates, maintained, table, correct, gradable, stateful, installs, difference

Introduction
The rapid evolution of Large Language Models (LLMs) has initiated a profound shift in how software
is specified, implemented, and maintained [ 1,2]. AI-assisted coding tools such as Cursor and Codex
∗Equal contribution.
†Chao Huang is the Corresponding Author.
Preprint. Under review.arXiv:2512.07921v1  [cs.SE]  8 Dec 2025

## Page 2

1. The Aspiration:
Document -to-Repository
AI Agents Human Experts~42%~72%Paper
Replication scoreThe Reality:
A Major Performance Gap2. The Core Obstacle:
Info Overload vs. Context Bottleneck
Information
Overloa d
LLM Context
Bottleneck
1. Specification
Preservation2. Global
Consistenc y3. Underspecified
Desig n4. Executable
Faithfulnes sThis leads to four key failures3. The Solution: DeepCode's
Information -Flow Management
Result: Surpasses Human Expert
Cursor Human Experts DeepCode
Four Information Operations
1. Source
Compression2. Structured
Indexing3. Knowledge
Injection4. Error
Correction
Blueprint CodeMem CodeRAG Verification
Maximizing relevant signalsFigure 2: From Challenge to Solution of DeepCode. Left: Current AI agents achieve only a 42%
paper replication score compared to 72% for human experts, highlighting the limitations of existing
agents. Middle: The core challenge stems from information overload conflicting with LLM context
limits, causing four key failure modes. Right: DeepCode addresses this through four information
operations (Blueprint, CodeMem, CodeRAG, Verification), surpassing human expert performance.
have already transformed everyday development practice by automating routine implementation tasks
and offering intelligent inline suggestions [ 3,4]. Yet these systems remain fundamentally assistive:
they operate at the level of code completion, assuming that a human engineer still performs the higher-
level tasks of understanding specifications, planning system architecture, and validating behavior.
Recent advances in agentic LLM frameworks point toward a more ambitious paradigm—what we
termagentic software engineering—in which LLM-based agents are expected to plan, orchestrate, and
refine entire software projects from high-level natural language or document-level specifications [ 5,6].
In this emerging regime, programming shifts fromwriting codetowriting specifications, and the
central question becomes:can an artificial coding agent behave as an autonomous engineer that
translates rich, informal specifications into comprehensive, robust systems?
A natural and stringent testbed for this paradigm ishigh-fidelity, document-grounded program
synthesis, where a complex scientific paper serves as the sole specification and the goal is to
produce a fully executable implementation that faithfully reflects it. Such papers are detailed
multimodal specifications, combining informal exposition with equations, pseudo-code, and scattered
hyperparameters. In this work, we tackle the highly challenging task of reproducing machine
learning papers as complete code repositories. Recent efforts have explored this via LLM-based
agents. PaperBench evaluates frontier models on 20 ICML papers, finding the strongest model (o1)
with IterativeAgent achieves only 42.4% replication score, far below 72.4% for human experts [ 7].
PaperCoder employs a multi-agent pipeline spanning planning, analysis, and generation, reaching
51.14% reproduction rate on PaperBench [ 8]. These modest results reveal that current approaches fall
well short of reliable, end-to-end replication. We identify four key challenges that underlie this gap:
(i) Specification Preservation.Papers describe the target system through scattered, multimodal
constraints. Preserving a faithful mapping from this fragmented specification to implementation is
inherently difficult.(ii) Global Consistency under Partial Views.Repositories comprise interde-
pendent modules, but generation proceeds file-by-file under limited context. Maintaining consistency
across interfaces, types, and invariants under finite context windows easily leads to broken abstrac-
tions.(iii) Completion of Underspecified Designs.Papers specify only algorithmic cores, leaving
implementation details and experimental frameworks implicit. Inferring these consequential but
underspecified choices is non-trivial.(iv) Executable Faithfulness.Faithful reproduction requires
executable systems, not just plausible code. Long-horizon generation often yields repositories with
subtle logic bugs, dependency conflicts, and fragile pipelines that prevent end-to-end execution.
We argue that fundamentally addressing these challenges requiresprincipled information-flow man-
agement. We abstract the synthesis process as the transmission of a high-entropy specification—the
scientific paper—through a sequence of bandwidth-constrained channels, defined by the LLM’s
2

## Page 3

context windows. Naive strategies that simply concatenate raw documents with growing code history
induce channel saturation, where redundant tokens mask critical algorithmic constraints, causing
the effective Signal-to-Noise Ratio to collapse. Consequently, valid repository generation requires a
paradigm shift governed bycontextual information maximization: at each generation step, the system
must actively maximize the density of task-relevant signals while suppressing irrelevant noise.
Motivated by this perspective, we introduceDeepCode, an open agentic coding framework that
fundamentally reimagines repository-level synthesis as a problem ofhierarchical information-flow
management. Rather than treating synthesis as a monolithic process, DeepCode systematically ad-
dresses the doc-to-repos challenges by instantiating the proposed paradigm through four orchestrated
information operations: (1)source compression, which distills unstructured multi-modal specifica-
tions into a precise structural blueprint to maximize signal density; (2)structured indexing, which
abstracts the evolving repository state into concise memory entries to maintain global consistency
without context saturation; (3)conditional knowledge injection, which leverages retrieval-augmented
generation to bridge implicit specification gaps with standard implementation patterns; and (4)error
correction, which utilizes closed-loop verification to transform execution feedback into corrective
signals for rectifying transmission errors. Our contributions are threefold:
•We characterize the task of high-fidelity document-to-repository synthesis through an information-
theoretic lens, identifying the central conflict as aninformation-overload vs. context-bottleneck
conflict. From this perspective, we propose an information-theoretic design principle: effective
agentic coding systems must explicitly structure, route, and compress information to maximize
task-relevant signal under finite context budgets.
•We instantiate this principle in DeepCode, a systematic framework that orchestrates four strategic
information operations: blueprint distillation, stateful memory management, conditional knowledge
injection, and closed-loop verification. By dynamically optimizing the signal-to-noise ratio within
the context window, DeepCode effectively resolves the challenges of long-range specification
preservation, cross-file consistency, and implicit knowledge gaps in complex generation tasks.
•Extensive evaluations on the PaperBench benchmark demonstrate that DeepCode achieves state-of-
the-art performance, decisivelyoutperforming leading commercial agents(e.g. Cursor, Claude
Code, Codex) and, notably,surpassing human expert performanceon key reproduction metrics.
Furthermore, our analysis reveals that principled information-flow management yields significantly
larger performance gains than merely scaling model size or context length, offering a pivotal
direction for the future design of autonomous software engineers.
2 Preliminary
2.1 Task Definition
The primary objective of this work is to develop a system forhigh-fidelity program synthesis. We
formalize this as the process of learning a mapping function, Fgen, which transforms a specification
document,D, into a complete and executable code repository,P. The core function is defined as:
Fgen:D→P(1)
where Drepresents the space of specification documents and Prepresents the space of valid code
repositories. Such that for a given input document D ∈D , the output is a program repository
P=F gen(D). We address two primary manifestations of this task:
•Scientific Paper Reproduction:Given a scientific paper from domains such as machine learning
or computer sciences as the source document D, the system should generate the full source code P
required to replicate the paper’s key experiments and results.
•Software System Generation:Given a comprehensive technical design document or a concise
natural language requirement for a software application (e.g., specifying UI, backend APIs, and
database schema) as D, the system should generate the corresponding multi-component software
repositoryP, including frontend, backend, and configuration files.
Input: Source Document D.The source document Dis represented as a sequence of multi-modal
elements, D= (d 1, d2, . . . , d L), where each element dican be a block of text, a mathematical
3

## Page 4

equation, a table, a figure, or a snippet of pseudocode. The length Lof this sequence is typically
large, posing significant challenges for models with finite context windows.
Output: Code Repository P.The target output Pis not a single file but a structured repository. We
define it as a tuple:
P= (T,C,M)(2)
Here,Trepresents the directory structure that organizes the files in C.C={c 1, c2, . . . , c N}is a
set of Nsource code files. The generation of a coherent set Cwhere files correctly interact (e.g.,
via imports and function calls) is a non-trivial problem of ensuring cross-file consistency. Mis
the dependency manifest (e.g. requirements.txt ,package.json ,README.md file) specifying all
external libraries required to run the code.
2.2 Objectives
An ideal synthesis function Fgenmust generate a repository P∗that optimizes a composite scoring
function. Under our paradigm ofprincipled information-flow management, this optimization is
framed as maximizing the effective signal-to-noise ratio across the synthesis channel. The optimal
output is defined as:
P∗= arg max
P∈PScore(P|D)(3)
To overcome the conflict between information overload and finite context bandwidth, the scoring
function decomposes into four distinct objectives, each corresponding to an information operation:
•Specification Preservation:The repository must faithfully implement the rigid algorithmic
constraints hidden within the multimodal source document. The objective is to maximize signal
density by extracting precise blueprints from the unstructured input noise.
•Global Structural Consistency:The generated modules must maintain strict interface compatibil-
ity and type coherence. The objective is to maintain state consistency without context saturation,
achieved by indexing the evolving codebase into compact, retrievable summaries.
•Domain Knowledge Grounding:The system must bridge the gap between abstract academic
descriptions and concrete engineering implementations. The objective is to resolve underspecified
designs by conditionally injecting standard libraries and patterns from external knowledge bases.
•Functional Executability:The final repository must be robust and runnable. The objective is to
minimize transmission errors (bugs) by treating runtime execution feedback as a corrective signal
to iteratively refine the generated code.
Our framework is designed to satisfy these objectives by explicitly routing and compressing informa-
tion, enabling high-fidelity repository generation under strict context window constraints.
3 The DeepCode Framework
We introduce DeepCode, a multi-stage framework designed to instantiate the principle of principled
information-flow management for repository-level synthesis. To solve the optimization problem,
DeepCode decomposes the generation process into three orchestrated phases, each serving a distinct
information-processing role to maximize the effective signal-to-noise ratio. The process initiates
with(1) Blueprint Generation, where a planning agent acts as a source compression mechanism,
distilling the high-entropy source document Dinto a structured, high-signal implementation blueprint
to extract critical constraints while filtering narrative noise. Guided by this blueprint, the subsequent
(2) Code Generationphase synthesizes source files while preventing channel saturation through
two integrated mechanisms: a stateful Code Memory (CodeMem) that performs structured indexing
of the evolving codebase to maintain cross-file consistency, and a CodeRAG system that performs
conditional knowledge injection to bridge implicit domain gaps with standard implementation patterns.
Finally, the framework concludes with(3) Automated Verification, a closed-loop error correction
phase where a validation agent treats runtime execution feedback as corrective signals to identify and
rectify transmission errors, ensuring the functional correctness of the final output.
4

## Page 5

Phase 3: Automated Verification andRefinement
Static Analysis and Code Quality Refinement
Analysis Agent
Static Analysis of Code Issues
Modification Agent
Line-level Modifications inspired by LSP
Sandbox Execution and Functional Correction
Sandbox
 Refinement
Test data
Execution traceAnalyzing trace
LSP-based 
refinementOptimal Code
Repository
…
Source 
DocumentsHierarchical Content Segmentation
Structural Parsing
Keyword -chunk Pairs
…
Concept Agent Algorithm Agent
Conceptual Analysis Schema Algorithmic Implementation Schema
Planning Agent
Phase 1: Blueprint GenerationCoding Blueprint
Project File Hierarchy Component Specification
Verification Protocol Execution EnvironmentStaged
Dev.
PlanCode Files
Phase 2: Code GenerationLLMs
GenerateIterative Code Generation
CodeRAG CodeMemAdditional
Resources
Retrieved
Knowledge
Target
Code FileMemory
Context
Memory
SummarizationUpdate New memory entry
Next IterationPick from 
Blueprint
Paper
Docs
Passage 
Figure 3: The overall framework of DeepCode.
3.1 Phase 1: Blueprint Generation
The primary goal of the first phase is to perform source compression: distilling the unstructured,
lengthy content of a source document (e.g. a scientific paper) into a structured, machine-readable
implementation blueprint. This distillation process directly mitigates the challenges of information
overload by transforming the raw input Dinto a high-density signal format. The process begins with
a crucial preprocessing step: hierarchical content segmentation.
3.1.1 Hierarchical Content Segmentation
Instead of feeding the entire document Dinto an LLM, we first parse it into a structured representation
that facilitates targeted information access. We introduce ahierarchical content index, which
leverages the inherent structure of academic papers and technical documents. The process is:
1.Structural Parsing:The source document Dis parsed to identify its hierarchical structure based
on explicit delimiters like section and subsection titles (e.g. "3. Methodology", "3.1. Model
Architecture"). This divides the document into a set of content chunksS={s 1, s2, . . . , s K}.
2.Keyword-Chunk Association:Each chunk skis stored as a key-value pair (hk, ck), where the
heading hkserves as a natural, high-level semantic keyword, and ckis the corresponding raw text
content of that section.
This indexed structure effectively transforms the problem from one of long-context comprehension
to a series of more manageable, on-demand retrievals. An agent no longer needs to process the
entire document at once. Instead, it can query the index using semantic keywords (e.g. requesting the
content associated with "Model Architecture") to fetch only the most relevant context for its current
task. This approach drastically reduces the token load for any single operation and allows the model
to focus its limited context window on the most pertinent information, thereby solving the problem of
context overload and information forgetting. This structured representation serves as the foundational
input for the specialized agents that perform the detailed analysis in the subsequent steps.
3.1.2 Multi-Agent Specification Analysis
Following the hierarchical segmentation, we employ a specialized multi-agent system to conduct
a deep and structured analysis of the document’s content. This approach decomposes the complex
comprehension task into two parallel tracks, executed by aConcept Agentand anAlgorithm Agent.
Each agent is equipped with a specific prompt and interacts with the indexed document to extract
complementary layers of information, ensuring a comprehensive understanding without processing
the entire document simultaneously.
5

## Page 6

Concept Agent: High-Level Structural and Conceptual Mapping.The Concept Agent is tasked
with building a holistic, high-level understanding of the document. Its primary objective is to map the
paper’s entire conceptual structure, identify its core scientific contributions, and outline the necessary
components for a successful experimental reproduction. Operating on the indexed document, the
agent is instructed to use a segmented reading strategy, querying the index with semantically broad
keywords (e.g. “introduction”, “method”). This allows it to assemble a comprehensive overview by
strategically fetching relevant sections. The output of this agent is a structuredConceptual Analysis
Schema. This schema comprises a detailed paper structure map, a method decomposition map
outlining the system’s core functional components, an implementation map aligning claims with code
requirements, and a reproduction roadmap specifying the criteria for success. Collectively, these
elements translate the paper’s narrative into a structured project plan.
Algorithm Agent: Low-Level Technical Detail Extraction.Complementing the conceptual
overview, the Algorithm Agent is responsible for the meticulous extraction of every low-level
technical detail required for an exact implementation. It’s designed to perform an exhaustive search
for all algorithms, mathematical formulations, model architectures, training procedures, and hy-
perparameters. Moreover, it can leverage online search capabilities to retrieve relevant algorithm
implementations from the web as references. Like the Concept Agent, it leverages the segmented read-
ing strategy but uses a distinct set of highly specific keywords (e.g. “algorithm”, “hyperparameter”) to
perform targeted queries on the most technically dense sections of the document. The agent’s output
is a granularAlgorithmic Implementation Schema. This schema captures verbatim pseudocode from
algorithm boxes, exact mathematical equations and their variables, detailed layer-by-layer network
architectures, and a comprehensive list of all hyperparameters with references to their locations in the
paper. This schema serves as a precise, unambiguous technical specification, designed to leave no
detail to interpretation during the code generation phase.
3.1.3 Synthesizing the Implementation Blueprint
The analytical outputs from the Concept and Algorithm agents are then synthesized by theCode
Planning Agentinto a single, holistic implementation blueprint. This agent’s critical function
is to orchestrate the high-level conceptual framework with the low-level technical specifications,
performing a final disambiguation and grounding step. It reconciles the architectural overview with
the granular implementation details, ensuring that every abstract component is directly linked to a
precise technical specification. Should any inconsistencies arise, the agent is authorized to perform
targeted queries on the indexed document to resolve them. The finalImplementation Blueprint Bis
a structured intermediate representation designed to be a self-contained, unambiguous specification
for code generation. This blueprint is organized into the following canonical sections:
•Project File Hierarchy:A prioritized project file structure that dictates the logical organization of
the codebase and the implementation order of its modules.
•Component Specification:A granular specification for every module, class, and function, explic-
itly mapping each to its corresponding algorithmic pseudocode and mathematical formulation.
•Verification Protocol:A formal plan for validating the final implementation. It defines the
experimental setup, specifies the target metrics from the source document, and establishes the
success criteria for reproduction.
•Execution Environment:A complete specification of all software dependencies, library versions,
and requisite hardware configurations needed to compile and run the code.
•Staged Development Plan:A phased implementation roadmap that defines the build order of
components and integrates staged verification checks to ensure modular correctness.
By consolidating all distilled information into this canonical blueprint, the Code Planning Agent
concludes the specification distillation phase. This artifact serves as the definitive "source of truth" for
the subsequent code generation phase, effectively resolving the long-context challenge by providing
a dense, structured, and actionable input that obviates any need for the coding agents to interact with
the original, lengthy document.
3.2 Phase 2: Code Generation
Upon generating the high-signal blueprint, the second phase synthesizes the code repository. This
phase maximizes the density of relevant context while preventing channel saturation caused by the
6

## Page 7

accumulation of raw code history. A naive iterative approach, which appends previously generated
code to the prompt, leads to a collapse in the signal-to-noise ratio and induces hallucinations. To
overcome this, we propose a dual-mechanism strategy for efficient information routing: (1) a stateful
CodeMemthat performs structured indexing of the evolving repository to maintain internal structural
cohesion without context bloat, and (2) aCodeRAGsystem that performs conditional knowledge
injection, grounding the implementation in external patterns to bridge implicit knowledge gaps.
3.2.1 Stateful Generation with CodeMem
The core of our generation process is the Code Memory mechanism, a strategy designed to maintain a
compressed, structured representation of the repository’s state, thereby ensuring cross-file consistency
without suffering from prohibitive context lengths. Instead of passing the full source code of
previously implemented files to the generative agent, we iteratively build and query a structured
memory bank,M.
Let the set of all files to be implemented, as defined by Sec. 2, be C={c 1, c2, . . . , c N}. The
generation process is an iterative loop over t= 1, . . . , N . At each step t, we maintain the set of
implemented files, Ct−1, and the set of unimplemented files, Ut−1. The process for generating the
target file for the current step,ˆc t, is as follows:
1.Context Formulation.The generation context for the current step, Xt, is constructed not from
raw source code, but from the static implementation blueprint Band a dynamically selected subset
of the Code Memory, Mt−1. The agent first identifies which previously implemented files are
relevant to the current target file ˆct(where ˆctdenotes the blank code file to be generated, and ct
denotes the resulting generated code file). It then retrieves only their corresponding summaries
from the memory bank:
Xt= (B,SelectRelevantMemory(M t−1,ˆct))(4)
where SelectRelevantMemory is a function that queries Mt−1to fetch only the essential sum-
maries of dependencies.
2.Code Generation.The coding agent, represented by the LLM function L, synthesizes the source
code for the target file based on the curated context:
ct=L(X t)(5)
3.Memory Update.After generating the code ct, the system clears the generation context. A
specialized summarization agent, S, is then invoked. This agent analyzes the newly generated
source code ctto extract its structural essence and create a new memory entry, mt. The Code
Memory is then updated:
Mt=M t−1∪ {m t}(6)
The summarization agent Sdistills the code into a structured format that captures all information
necessary for inter-module communication. Each memory entry mtis a structured object containing:
•Core Purpose ( Pt):A concise, natural language summary of the file’s primary responsibility and
role within the repository.
•Public Interface ( It):A formal description of all externally accessible classes, functions, and
constants, including their signatures and purposes (e.g., Class(params): methods).
•Dependency Edges ( Et):A comprehensive map of the file’s position within the project’s depen-
dency graph. This structured entry specifies bothafferent couplings(internal dependencies),
detailing the specific imports from other project modules and external packages, and predictedef-
ferent couplings(external dependencies), identifying which unimplemented modules are expected
to consume this file’s public interface.
•Next Implementation Target ( ˆct+1):A decision on the next file to be implemented, based on the
blueprint, dependency graph and the current state. Note that, to avoid introducing noise into the
memory, this information is separated fromm tand provided independently as part ofLinput.
This mechanism effectively decouples the context size from the repository size. The context provided
to the agent at any step tremains compact, containing only the high-level blueprint and the highly
compressed summaries of relevant, already-implemented files. This stateful, summary-based ap-
proach allows our system to maintain global consistency and logical cohesion across a large number
of files, directly solving the long-context and cross-file consistency challenges.
7

## Page 8

3.2.2 Knowledge Grounding with CodeRAG
While the Code Memory mechanism ensures internal consistency, it does not address the challenges
of model hallucination or the omission of implicit domain knowledge. To mitigate these issues, we
introduce a retrieval-augmented generation framework,CodeRAG, which grounds the synthesis
process in a pre-indexed corpus of relevant, high-quality code repositories. This process is divided
into two stages: an indexing phase and an adaptive retrieval phase during code generation.
Repository Indexing.The goal of this phase is to analyze a set of relevant source code repositories,
R={R 1, R2, . . . , R K}, and build a structured, queryable index, J. The process, modeled by
Iindex:R × B → J, consists of the following steps:
1.Relevance Filtering:For each repository Rk∈ R, we perform an initial LLM-based filtering
to identify a subset of source files, C′
k⊂R k, that are most relevant to the target project structure
defined in the implementation blueprint B. In this context, Rcan denote either the corresponding
repository cited in the references of the target paper or other relevant repositories identified through
online search. This focuses computational resources on the most promising assets.
2.Code Understanding:Each relevant source file c′
s∈ C′
kis independently analyzed to create
a structured summary, analogous to the memory entries described previously. This summary
captures the file’s purpose, key concepts, and public interfaces.
3.Relationship Mapping:The core of the indexing process is to establish explicit links between
the analyzed source files and the target files in our blueprint. For each source file summary, an
agent maps it to one or more target files inB, generating a set of relationship tuples.
The final output index Jis a structured knowledge base containing a collection of relationship
tuples. Each tuple is defined as (c′
s,ˆct, τ, σ, γ) . Here, c′
sis a file in the source repository and ˆctis
the corresponding target file in the blueprint’s structure. τdenotes the relationship type, indicating
the nature of the potential contribution, while σis a confidence score representing the strength of
the mapping. γis a set of actionable context, such as helpful code snippets, usage suggestions, and
implementation patterns.
Adaptive Retrieval.During the iterative code generation phase, our framework will optionally query
the CodeRAG index Jto augment its context. At each generation step tfor a target file ˆct, the agent
makes an adaptive decision on whether to retrieve external knowledge. This decision is modeled by a
binary functionδ:
rt=δ(X t,ˆct)(7)
where flag rt∈ {0,1} andXtis the standard context containing the blueprint and relevant code
memory. The decision is based on the complexity of the target file and the level of detail available in
the blueprint. If rt= 1, the agent queries the index Jto find the most relevant relationship tuples for
ˆct. The retrieved context γfrom the highest-confidence relationship is used to create an augmented
context,X′
t:
X′
t=Xt∪ {Retrieve(J,ˆc t)}(8)
The final code is then generated using this enriched context: ct=L(X′
t). By dynamically incorpo-
rating proven implementation patterns from existing repositories, CodeRAG significantly reduces
the likelihood of generating erroneous or suboptimal code, thus bridging the knowledge gap for the
generative agent.
3.3 Phase 3: Automated Verification and Refinement
The final phase serves as an error correction mechanism to ensure the functional faithfulness of the
synthesized repository P. Recognizing that purely generative processes are prone to transmission
errors—manifesting as logic bugs, invalid dependencies, or dead code—this phase establishes a
crucial closed-loop feedback system absent in standard models. By treating execution outcomes as
corrective signals, the framework systematically identifies and rectifies defects through two sequential
stages: (1) a static analysis pass to ensure structural integrity and code quality, and (2) a dynamic
execution pass within a sandboxed environment to enforce functional correctness.
3.3.1 Static Analysis and Code Quality Refinement
The first stage addresses issues that can be detected without executing the code. This process is
orchestrated by a dedicated Analysis Agent and a Modification Agent.
8

## Page 9

Static Analysis.An Analysis Agent, denoted by the function Astatic, inspects the generated repository
Pagainst the implementation blueprint B. It produces a structured static analysis report, Rstatic,
which identifies a set of issues. This process can be formalized as:R static=A static(P,B).
The identified issues I={i 1, i2, . . . , i K}fall into two categories: i)Structural Discrepancies:This
includes integrity violations such as missing files specified in the blueprint or empty (zero-byte)
source files that were not correctly generated. ii)Code Quality Deficiencies:The agent leverages
an LLM to perform a quality assessment of each source file, assigning a quality score, q(ci), and
flagging sections with poor style, complexity, or maintainability.
Code Refinement.The report Rstaticis then passed to a Modification Agent, Amodify . This agent iter-
ates through each issue ik∈Iand applies a targeted fix. To perform precise, line-level modifications
without rewriting entire files, the agent utilizes a programmatic interface inspired by the Language
Server Protocol (LSP). We model this refinement operation as a function ΦLSPthat takes a file ciand
a modification instruction from the report, producing a corrected file c′
i. The overall process yields a
statically refined repositoryP′as:P′=A modify(P,R static).
3.3.2 Sandbox Execution and Functional Correction
After static refinement, the repository P′undergoes dynamic testing in a secure, isolated sandbox
environment to ensure it runs as intended.
Environment Verification and Setup.A Sandbox Agent, Asandbox , first validates the environment
setup instructions (e.g., in README.md ) against the dependencies specified in the blueprint B. Any
discrepancies are corrected. The agent then automatically provisions the specified environment and
installs all dependencies.
Iterative Execution and Correction.The agent then attempts to execute the main entry points of
the repository, using automatically generated test data and test files designed to exercise the core
algorithms and functions. The execution process, Esandbox , takes the repository P′
jat iteration j
(initiallyP′
0=P′) and produces an execution trace,T j, containing all outputs and error messages.
Tj=E sandbox (P′
j)(9)
This initiates an iterative refinement loop. If the trace Tjcontains errors ( Terror
j̸=∅), the Sandbox
Agent analyzes the error messages to identify the likely faulty files and the nature of the bug. It then
generates a modification instruction and invokes the LSP-based refinement function ΦLSPto patch the
code, producing the repository for the next iteration, P′
j+1. This loop continues until the execution is
successful or a maximum number of iterations is reached.
P′
j+1= Φ LSP(P′
j,Terror
j)(10)
The final verified output of our entire framework is the repository P∗=P′
J, where Jis the terminal
iteration of the refinement loop. This multi-stage verification and correction process ensures that the
synthesized code is not only structurally sound but also functionally correct and conformant to the
original specification.
4 Experiments
In this section, we evaluate the effectiveness of the proposed DeepCode framework by addressing
the following 3 research questions:RQ1:How does DeepCode perform compared to existing agent
frameworks?RQ2:How does the choice of different LLMs affect the performance of DeepCode?
RQ3:What is the contribution of each module within the DeepCode architecture?
4.1 Experiments Settings
Datasets.To evaluate DeepCode’s capabilities in code comprehension and generation, particularly
for automated vulnerability detection, we employPaperBench Code-Dev, an innovative benchmark
created by OpenAI [ 7]. PaperBench Code-Dev assesses AI models’ ability to independently reproduce
leading ML research from major conferences like ICML 2024, focusing on 20 significant papers.
Models are required to generate all necessary code from scratch, using only the research papers as
references, without accessing existing codebases from the original authors. These tasks are performed
9

## Page 10

in a virtual machine environment, with the goal of building a functional codebase, replicating
experiments, and creating a reproduce.sh script for execution. Each paper is accompanied by a
detailed evaluation rubric approved by the authors, which breaks down the reproduction task into 8,316
specific, gradable components, meticulously assessed using a hierarchical weighting scheme and
SimpleJudge, a sophisticated automated judge powered by OpenAI’s o3-mini model. This benchmark
is rigorously crafted to challenge AI with tasks requiring advanced natural language understanding,
algorithmic reasoning, and the ability to generate reliable code from abstract descriptions, all of
which are crucial skills for automating vulnerability detection effectively.
Baselines.In order to evaluate the effectiveness of the proposed framework, we include a range of
baseline methods for comparison. These baselines fall into four distinct categories:
(1) LLM Agents.We compare against results reported in [ 7] for several state-of-the-art language
models using two agent scaffolding approaches: (1)BasicAgent, a simple tool-use loop based on
Inspect AI’s basic agent that allows models to terminate early, and (2)IterativeAgent, which forces
models to use their full allocated time and employs prompts designed to encourage incremental,
piecemeal progress. All agents run in Ubuntu 24.04 Docker containers with access to a single A10
GPU, the internet, and standard development tools including bash, Python, web browsing, and file
reading capabilities [ 7]. The baseline models include GPT-4o, o1, o3-mini, DeepSeek-R1, Claude
3.5 Sonnet, and Gemini 2.0 Flash, with most experiments using a 12-hour time limit (extended to 36
hours for select o1 runs).
(2) Scientific Code Agents.PaperCoder[ 8]. PaperCoder (also referred to as Paper2Code) is a multi-
agent LLM framework that transforms machine learning papers into executable code repositories via
a three-stage pipeline: planning, which constructs implementation roadmaps, system architecture
diagrams, and file dependencies; analysis, which extracts file-level implementation details; and
generation, which produces modular code in dependency order.
(3) Commercial Code Agents.We compare against three state-of-the-art commercial code agents
that provide AI-powered development assistance through different interfaces and capabilities:
•Cursor(Version 1.7.52) is an AI-assisted integrated development environment built as a fork of
Visual Studio Code with additional AI features. Cursor allows developers to choose between
cutting-edge LLMs and provides codebase embedding models that give agents deep understanding
and recall [ 9]. In our experiments, Cursor uses Claude Sonnet 4.5-thinking as the underlying model.
•Claude Code(Version 2.0.22) is Anthropic’s agentic coding tool that lives in the terminal and
helps developers turn ideas into code. Claude Code maintains awareness of the entire project
structure, can find up-to-date information from the web, and with MCP can pull from external
data sources like Google Drive, Figma, and Slack. It can directly edit files, run commands, create
commits, and use MCP to read design docs or update tickets [ 10]. Our evaluation uses Claude
Sonnet 4.5-thinking.
•Codex(Version codex-cli 0.47.0) is OpenAI’s coding agent that runs locally from the terminal
and can read, modify, and run code on the user’s machine. Codex is optimized for use with
GPT-5-Codex for agentic coding, with configurable reasoning levels from medium to high for
complex tasks. In auto approval mode, Codex can read files, make edits, and run commands in the
working directory automatically [11]. We configure Codex with GPT-5 Codex-high.
(4) Human Experts.The human baseline [ 7] consists of 8 ML PhD students and graduates from top
institutions (e.g. Berkeley, Cambridge, Carnegie Mellon) who worked part-time over a four-week
window on a 3-paper subset (all-in-one, fre, stay-on-topic). Participants had similar computational
resources (A10 GPU) and could use AI coding assistants like ChatGPT and GitHub Copilot. The
best-of-3 human attempts (Best@3) represent expert-level performance on this subset.
Experimental Setup.To evaluate DeepCode’s efficacy in high-fidelity repository synthesis, we adopt
a rigorous framework under realistic constraints. The setup combines a secure execution environment
and the PaperBench protocol for fair, reproducible, detailed comparisons across baselines.
(1) Implementation Environment.All experiments are conducted within an Ubuntu 22.04 LTS-
based sandboxed environment. This infrastructure is provisioned with a standard Python development
stack and essential dependencies. DeepCode is configured to operate within this isolated space,
retaining privileges for file system manipulation, shell command execution, and internet access,
thereby simulating a standard software research and development workflow.
10

## Page 11

(2) Task Execution.DeepCode accepts the target paper in both PDF and Markdown formats, along
with any supplementary addenda, as primary inputs. To ensure that generated solutions stem from
algorithmic reasoning rather than retrieval, a source code blacklist is enforced during execution. This
protocol precludes access to the authors’ original repositories and known third-party implementations
during web browsing. With input parameters defined and the search space constrained, DeepCode
initiates its autonomous workflow for code generation and debugging.
(3) Grading Methodology.Assessment of the generated code follows the PaperBench Code-Dev
protocol, which focuses on structural and functional correctness and does not include post-submission
reproduction. Grading is carried out by SimpleJudge, an automated system based on OpenAI’s
o3-mini, which performs static analysis of the submitted repository against a set of fine-grained,
hierarchical criteria co-developed with the authors of the source paper. The judging logic is restricted
to the “Code Development” leaf nodes of this rubric and examines core aspects of software quality,
including static correctness (syntax validity and compliance with language standards), dependency
validity (completeness and correctness of dependency specifications such as requirements.txt ),
project structure (coherent and consistent organization of files and directories), and algorithmic
fidelity (faithful implementation of the algorithms and interfaces described in the original paper).
This procedure is designed to align the evaluation with the central technical contributions of the work.
(4) Evaluation Metrics and Protocol.Our primary evaluation metric is the Replication Score, which
quantifies the proficiency of DeepCode in translating theoretical concepts into a functional codebase.
The score for a single replication trial is derived from the hierarchical rubric through a bottom-up
aggregation process.(i) Leaf node scoring:SimpleJudge first evaluates each leaf node criterion
on a binary basis, assigning a score of 1 for “pass” (compliance) and 0 for “fail” (non-compliance).
(ii) Score aggregation:The score for any parent node is then computed as the weighted average of
the scores of its immediate children. The weights, predetermined during the rubric design, reflect
the relative importance of each sub-task.(iii) Final score derivation:This recursive aggregation
continues up the hierarchy until a single score is obtained for the root node, which serves as the
Replication Score for that trial.
To account for the stochasticity inherent in code generation, we adopt a strict evaluation protocol. For
each target paper, three independent replication trials are performed, and each resulting repository is
scored separately by SimpleJudge using the procedure described above. The final Replication Score
is the average of the three scores, mitigating outliers and providing a more stable and reliable measure
of the model’s typical performance.
4.2 Main Results
The primary results of our experiments are detailed in Figure 4. We analyze the performance
of DeepCode against the four established categories of baselines: general-purpose LLM agents,
specialized scientific code agents, commercial code agents, and human experts.
•Comparison against LLM Agents.Figure 4 presents average replication scores across all
benchmark papers. Among general-purpose LLM agents, performance varies significantly by model
and scaffolding. With BasicAgent, Claude-3.5-Sonnet achieves the highest score (35.4 ±0.8), while
other frontier models range from 5.0 to 19.5. IterativeAgent scaffolding improves some models,
with o1 reaching the best LLM agent performance of 43.3 ±1.1. DeepCode achieves 73.5 ±2.8,
/uni0000002b/uni00000058/uni00000050/uni00000044/uni00000051/uni00000003/uni00000028/uni0000005b/uni00000053/uni00000048/uni00000055/uni00000057/uni00000027/uni00000048/uni00000048/uni00000053/uni00000026/uni00000052/uni00000047/uni00000048/uni00000013/uni00000015/uni00000018/uni00000018/uni00000013/uni0000001a/uni00000018/uni00000014/uni00000013/uni00000013
/uni0000001a/uni00000015/uni00000011/uni00000017/uni00000008/uni0000001a/uni00000019/uni00000011/uni0000001a/uni00000008/uni00000014/uni00000011/uni00000003/uni0000002b/uni00000058/uni00000050/uni00000044/uni00000051/uni00000003/uni00000028/uni0000005b/uni00000053/uni00000048/uni00000055/uni00000057/uni00000003/uni0000000b/uni00000037/uni00000052/uni00000053/uni00000003/uni00000030/uni0000002f/uni00000003/uni00000033/uni0000004b/uni00000027/uni0000000c
/uni00000026/uni00000052/uni00000047/uni00000048/uni0000005b
/uni00000026/uni0000004f/uni00000044/uni00000058/uni00000047/uni00000048/uni00000003/uni00000026/uni00000052/uni00000047/uni00000048/uni00000026/uni00000058/uni00000055/uni00000056/uni00000052/uni00000055
/uni00000027/uni00000048/uni00000048/uni00000053/uni00000026/uni00000052/uni00000047/uni00000048/uni00000013/uni00000015/uni00000018/uni00000018/uni00000013/uni0000001a/uni00000018/uni00000014/uni00000013/uni00000013
/uni00000017/uni00000013/uni00000011/uni00000013/uni00000008/uni00000018/uni0000001b/uni00000011/uni0000001a/uni00000008 /uni00000018/uni0000001b/uni00000011/uni00000017/uni00000008/uni0000001b/uni00000018/uni00000011/uni00000017/uni00000008/uni00000015/uni00000011/uni00000003/uni00000026/uni00000052/uni00000050/uni00000050/uni00000048/uni00000055/uni00000046/uni0000004c/uni00000044/uni0000004f/uni00000003/uni00000026/uni00000052/uni00000047/uni00000048/uni00000003/uni00000024/uni0000004a/uni00000048/uni00000051/uni00000057/uni00000056
/uni00000033/uni00000044/uni00000053/uni00000048/uni00000055/uni00000003/uni00000026/uni00000052/uni00000047/uni00000048/uni00000055/uni00000027/uni00000048/uni00000048/uni00000053/uni00000026/uni00000052/uni00000047/uni00000048/uni00000013/uni00000015/uni00000018/uni00000018/uni00000013/uni0000001a/uni00000018/uni00000014/uni00000013/uni00000013
/uni00000018/uni00000014/uni00000011/uni00000014/uni00000008/uni0000001a/uni00000016/uni00000011/uni00000019/uni00000008/uni00000016/uni00000011/uni00000003/uni00000036/uni00000046/uni0000004c/uni00000048/uni00000051/uni00000057/uni0000004c/uni00000049/uni0000004c/uni00000046/uni00000003/uni00000026/uni00000052/uni00000047/uni00000048/uni00000003/uni00000024/uni0000004a/uni00000048/uni00000051/uni00000057
/uni0000002a/uni00000048/uni00000050/uni0000004c/uni00000051/uni0000004c/uni00000010/uni00000015/uni00000011/uni00000013/uni00000010/uni00000049/uni0000004f/uni00000044/uni00000056/uni0000004b/uni0000002a/uni00000033/uni00000037/uni00000010/uni00000017/uni00000052
/uni00000027/uni00000048/uni00000048/uni00000053/uni00000036/uni00000048/uni00000048/uni0000004e/uni00000003/uni00000035/uni00000014/uni00000052/uni00000016/uni00000010/uni00000050/uni0000004c/uni00000051/uni0000004c
/uni00000026/uni0000004f/uni00000044/uni00000058/uni00000047/uni00000048/uni00000003/uni00000016/uni00000011/uni00000018/uni00000003/uni00000036/uni00000052/uni00000051/uni00000051/uni00000048/uni00000057/uni00000052/uni00000014
/uni00000027/uni00000048/uni00000048/uni00000053/uni00000026/uni00000052/uni00000047/uni00000048/uni00000013/uni00000015/uni00000018/uni00000018/uni00000013/uni0000001a/uni00000018/uni00000014/uni00000013/uni00000013
/uni00000018/uni00000011/uni00000013/uni00000008/uni0000001a/uni00000011/uni0000001a/uni00000008/uni0000001c/uni00000011/uni0000001b/uni00000008/uni00000014/uni00000019/uni00000011/uni00000017/uni00000008/uni00000016/uni00000018/uni00000011/uni00000017/uni00000008/uni00000017/uni00000016/uni00000011/uni00000016/uni00000008/uni0000001a/uni00000016/uni00000011/uni00000019/uni00000008/uni00000017/uni00000011/uni00000003/uni0000002f/uni0000002f/uni00000030/uni00000010/uni00000025/uni00000044/uni00000056/uni00000048/uni00000047/uni00000003/uni00000024/uni0000004a/uni00000048/uni00000051/uni00000057/uni00000056
Figure 4: Comparison of DeepCode with four baseline categories: (1) human experts, (2) state-of-
the-art commercial code agents, (3) scientific code agents, and (4) LLM-based agents
11

## Page 12

representing a 70% relative improvement over the best LLM agent baseline. This substantial gap
demonstrates that our framework’s specialized design, which incorporates systematic planning,
structured code generation and automated verification, provides significant advantages over general-
purpose agent scaffolding.
•Comparison against Scientific Code Agents.PaperCoder, a specialized multi-agent framework
designed for transforming machine learning papers into executable code, achieves a score of
51.1±1.4, outperforming all LLM agents baselines. However, DeepCode achieves a significantly
higher score of 73.5 ±2.8—an improvement of over 22 points. This substantial gain suggests that
our approach to task decomposition, code generation, and repository-level integration is markedly
more effective than existing specialized methods.
•Comparison against Commercial Code Agents.Table 1 details a direct comparison with
leading commercial agents on a 5-paper subset. DeepCode achieves an average score of 0.8482,
decisively outperforming Codex (0.3997), Cursor (0.5841), and Claude Code (0.5871). This result is
particularly noteworthy: DeepCode uses the same base model as both Cursor and Claude Code. The
dramatic performance difference provides strong evidence that our framework’s performance gains
are not merely a product of a powerful base model. Rather, the advantage is directly attributable to
the superior agentic architecture, planning, and execution strategies of DeepCode.
•Comparison against Human Experts.The most compelling finding is the comparison to human
expert performance. As shown in the final rows of Figure 4, we benchmarked performance on
the 3-paper subset. The human baseline, which represents the best-of-3 attempts from ML PhD
students, achieved a score of 72.4. Our DeepCode’s average performance on this same subset was
75.9±4.5, meaning it not only competes with but exceeds the score of the best attempt from a
human expert. This result strongly validates our approach, demonstrating its capability to automate
and even surpass expert-level performance on this highly challenging task.
Table 1: Reproduction scores of DeepCode and commercial code agents on 5-paper subset
Model fre rice bam pinn mech-u Avg.
Codex (GPT 5 Codex-high) 0.4095 0.3645 0.1937 0.5382 0.4926 0.3997
Claude Code (Claude Sonnet 4.5-think) 0.6286 0.3787 0.3829 0.7233 0.8222 0.5871
Cursor (Claude Sonnet 4.5-think) 0.6344 0.4186 0.3779 0.7748 0.7148 0.5841
DeepCode(Claude Sonnet 4.5-think)0.8435 0.7380 0.8530 0.9474 0.8888 0.8541
4.3 Analysis on Different LLMs
We evaluate DeepCode with five LLM backbones (Claude-4.5-Sonnet, GPT-5, Claude-3.5-Sonnet,
Gemini-2.5-Pro, DeepSeek-R1) on three PaperBench tasks (fre, all-in-one, stay-on-topic). The
tasks vary in specification complexity: fre and all-in-one contain long, interdependent setups with
overlapping constraints, while stay-on-topic provides more structured descriptions. Agent architecture
and tooling remain constant to isolate model capability effects.
As shown in Figure 5, reproduction scores exhibit consistent stratification across all three tasks.
Claude-4.5-Sonnet achieves the best or near-best performance (0.72-0.82), demonstrating particular
strength on fre and all-in-one where it more reliably reconstructs implementation details and multi-
stage pipelines implied by complex, underspecified descriptions. GPT-5 tracks Claude-4.5-Sonnet
closely on most metrics (0.69-0.81) and shows marginal advantages on stay-on-topic (0.81 vs.
0.72), suggesting additional robustness in maintaining alignment with fixed experimental framings,
though this does not overturn Claude-4.5-Sonnet’s overall dominance. Mid-tier models occupy an
intermediate performance range: Claude-3.5-Sonnet (0.48-0.57) and Gemini-2.5-Pro (0.44-0.73)
successfully recover main experimental skeletons but leave notable gaps in finer-grained procedural
steps. DeepSeek-R1 consistently underperforms ( ≈0.29), reproducing only fragments of target
workflows across all tasks. This stable ranking pattern across heterogeneous specifications indicates
that under fixed agent architecture, the underlying language model becomes the primary factor
determining the ceiling and reliability of automatic paper-level reproduction.
12

## Page 13

0.0 0.2 0.4 0.6 0.8 1.0DeepSeek-R1Gemini-2.5-proClaude-3.5-sonnetGPT-5Claude-4.5-sonnet
0.2930.7250.5200.7730.823fre
0.0 0.2 0.4 0.6 0.8 1.0
Replication Score0.2870.4400.5700.6940.758all-in-one
0.0 0.2 0.4 0.6 0.8 1.00.2930.5250.4800.8120.720stay-on-topicDeepSeek-R1 Gemini-2.5-pro Claude-3.5-sonnet GPT-5 Claude-4.5-sonnetFigure 5: DeepCode reproduction results on the 3-paper subset across LLM backbones
4.4 Ablation Studies
In this section, we conduct ablation studies on three core components of DeepCode: CodeRAG,
CodeMem, and Automated Verification. Specifically, we evaluate CodeRAG and Automated Verifica-
tion on a 3-paper subset (all-in-one, fre, stay-on-topic), while CodeMem is assessed on 5 randomly
selected tasks (test-time-model-adaptation, rice, mechanistic-understanding, fre, all-in-one). Our key
findings are summarized as follows.
(1) Impact of CodeRAG.To decouple the impact of CodeRAG, we conducted an ablation study using
Gemini-2.5-Flash. As visualized in Figure 6a, the integration of CodeRAG delivers a substantial
performance leap (up to 70% relative gain), effectively breaking the base model’s performance ceiling
(0.35–0.38). Notably, we observed negligible gains when applying CodeRAG to frontier models
like Claude 4.5 Sonnet. This contrast yields a critical insight: while reasoning giants likely encode
sufficient implementation patterns within their parameters, cost-efficient models like Flash suffer
from inherentknowledge gaps. Consequently, CodeRAG proves indispensable for these architectures,
acting as a vital bridge to fill implicit domain voids with standard practices—confirming that external
knowledge injection is essential for democratizing high-fidelity replication on lightweight models.
(2) Impact of CodeMem.We ablate CodeMem’s contribution on five PaperBench tasks using
Claude-4.5-Sonnet, comparing DeepCode’s structured memory against a "Simple" baseline that
naively evicts historical messages via sliding windows when approaching context limits.
Results demonstrate that unstructured eviction causes context saturation with signal loss: the Simple
protocol achieves only 0.33-0.43 in rice, fre, and mechanistic-understanding tasks due to dependency
truncation, where foundational class definitions are discarded before dependent code generation.
CodeMem’s structured indexing maintains task-relevant signal density, restoring scores to 0.70-0.92
by preserving critical dependencies without exhausting context budgets. Even in scenarios with strong
baseline performance (test-time-model-adaptation: 0.62 →0.72; all-in-one: 0.66 →0.76), Structured
memory delivers consistent gains, confirming our core thesis: effective agentic coding requires
explicit information flow management to maximize signal-to-noise ratio under context constraints.
(3) Impact of Automated Verification.Across 3 test papers, Automated Verification yields consistent
gains of 3.7–6.5%, elevating scores from 0.69–0.81 to 0.73–0.84. The layer primarily corrects three
types of residual errors: typos in variable names, missing dependencies, and wrong command-line
arguments. These errors prevent otherwise sound implementations from executing reliably. The
0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70
Replication Score (CodeRAG)fre
all-in-one
stay-on-topic+68.8%0.380 0.642
+41.0%0.354 0.499
+71.3%0.360 0.616- CodeRAG
CodeRAG
- Verification
Verification0.650 0.675 0.700 0.725 0.750 0.775 0.800 0.825 0.850Replication Score (Verification)
+3.7%0.8136 0.8435
+5.4%0.7193 0.7585
+6.5%0.6895 0.7342
(a) Ablation of CodeRAG and Verification
test-time-model
adaptationrice
mechanistic
understanding
fre
all-in-one0.20.40.60.81.0Simple
Code Memory (b) Ablation of CodeMem
Figure 6: Ablation studies of key components in DeepCode on PaperBench
13

## Page 14

modest improvement reflects an important fact: the earlier phases have already achieved technical cor-
rectness. Verification is a final pass to ensure reliable execution. It eliminates small but consequential
deviations that cause borderline implementations to fail, transforming them into faithful replications.
5