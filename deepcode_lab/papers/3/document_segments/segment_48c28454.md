# reference

**Content Type:** references
**Keywords:** development, open, modular, synthesis, time, indexes, triggering, deepcode, agentconducts, real

reference mining with
a global code memory, forming a closed-loop process that enforces repository-level consistency
during incremental generation.On the retrieval side, the Code Reference Mining and Code Indexing
agents implement a Retrieval-Augmented Generation (RAG) layer: they maintain multi-granularity
indices over a corpus of prior implementations and expose to the Code Generation agent semantically
relevant and structurally compatible code patterns, ranging from individual functions to reusable
design idioms. In parallel, the Code Memory agent maintains a structured representation of the
current repository state, including cross-file symbol tables, dependency graphs, and project-wide
conventions such as naming schemes, error-handling strategies, and configuration mechanisms.
Before emitting new code, the Code Generation agent issues queries to the Code Memory agent to
obtain the up-to-date repository context and applicable constraints; after generation, it writes back
the newly introduced symbols and dependencies, triggering an update of the global repository model.
This query–constraint–update loop allows DeepCode to align local synthesis decisions with global
architectural intent, reducing interface mismatches, naming drift, and latent coupling across the
codebase.
A.4 MCP Tool Stack in DeepCode
Table 5 summarizes the Model Context Protocol (MCP) tools integrated into DeepCode. The tools are
grouped into three functional categories:Perception & Retrieval,Cognitive Processing, andAction &
Execution. This organization makes the main stages of the system explicit. Perception & Retrieval
tools give the model access to up-to-date web search results, web pages, and binary documents such
as research papers and technical manuals, which helps mitigate the effects of the model’s knowledge
cut-off. Cognitive Processing tools then convert large codebases and long documents into semantic
indexes and context-window-compatible segments, so that the model can issue natural language
queries over existing artifacts and work with long technical materials. Action & Execution tools
finally operate on the local development environment by reading and writing project files, executing
shell commands, and interacting with the version control system.
Taken together, the tools in Table 5 form an end-to-end loop for assisted software development. The
system can retrieve external and local information, reorganize it into internal structures that fit within
the model’s context window, and then apply code changes while observing their effects through
commands such as tests or package installations. The table also shows that operations with side
effects on the environment (file I/O, command execution, and Git operations) are confined to the
Action & Executionlayer and are described as sandboxed and path-validated. This separation between
information access, semantic processing, and environment manipulation makes the extension of the
base language model through MCP tools transparent and easier to reason about.
21

## Page 22

Table 4: Functional Specifications of Specialized Sub-Agents in the DeepCode Framework
Agent Role Functional Description
Central Orchestrating
AgentFunctions as the central control unit, responsible for task decomposi-
tion, resource allocation, and the strategic coordination of sub-agents
based on the complexity of the input requirements.
Intent Understanding
AgentConducts semantic parsing of natural language inputs to extract
functional requirements, converting ambiguous user descriptions
into formal technical specifications.
Document Parsing Agent Processes unstructured technical documents (e.g., research papers).
It extracts multimodal information, including text, mathematical
formulas, and diagrams, to establish a ground truth for implementa-
tion.
Concept Analysis Agent Abstracts core theoretical concepts and logical flows from the parsed
specifications, ensuring the computational model aligns with the
theoretical underpinnings of the source material.
Algorithm Analysis Agent Evaluates and selects appropriate algorithmic strategies and data
structures. It focuses on optimizing computational complexity and
feasibility before code synthesis begins.
Code Planning Agent Formulates the software architecture and development roadmap.
This agent determines the technology stack, designs modular file
structures, and enforces design patterns to ensure scalability.
Code Reference Mining
AgentRetrieves external knowledge by identifying relevant open-source
repositories. It analyzes dependency graphs to recommend integra-
tion patterns and library usages.
Code Memory Agent Manages the state and context throughout the generation lifecycle.
It utilizes hierarchical data structures to retain historical decisions
and maintain semantic consistency across long-context interactions.
Code Generation Agent Synthesizes executable source code based on the architectural plan
and retrieved references. It implements functional interfaces and
integrates distinct modules into a cohesive codebase.
Automated Validation
AgentExecutes a rigorous quality assurance loop. It performs static analy-
sis, generates unit tests, and iteratively debugs the codebase to verify
functional correctness and adherence to specifications.
22

## Page 23

Table 5: Specification of Model Context Protocol (MCP) Tools Integrated into DeepCode. These
tools extend the Large Language Model’s capabilities across perception, cognitive processing, and
environment manipulation domains
Category MCP Server Name Functional Description & Academic Specifi-
cation
Perception & Retrievalbrave_search A real-time information retrieval interface lever-
aging the Brave Search API. It provides the
agent with temporal-aware access to web in-
dices, enabling the retrieval of up-to-date doc-
umentation and resolving knowledge cut-off
limitations.
bocha_mcp A specialized search module delivering struc-
tured "modal cards" and semantic summaries.
It serves as a secondary knowledge source, opti-
mizing token efficiency by returning structured
entities rather than raw HTML.
fetch A web content ingestion engine that retrieves
URL endpoints and normalizes heterogeneous
HTML structures into clean Markdown. It acts
as the agent’s primary reading interface for ex-
ternal documentation.
pdf_downloader Binary resource acquisition tool designed for
academic papers and technical manuals. It han-
dles HTTP streams to ingest non-textual doc-
ument formats (PDF/DOCX) for downstream
processing.
Cognitive Processingcode_reference_indexer A Retrieval-Augmented Generation (RAG)
module for local codebases. It constructs a
vector or semantic index of the project files,
allowing the agent to perform natural language
queries over the existing code structure.
document_segmentation A pre-processing utility implementing semantic
chunking algorithms. It partitions large techni-
cal documents into context-window-compliant
segments, facilitating the "Paper2Code" work-
flow for complex algorithm implementation.
Action & Executionfilesystem A sandboxed file I/O interface allowing con-
trolled read/write operations within the project
directory. It enforces path validation security
policies to prevent unauthorized system access
during code generation.
code_implementation The core generative engine encapsulated as an
MCP tool. It orchestrates the synthesis of func-
tional code blocks, integrating logic planning
with atomic file writing operations to ensure
code coherence.
command_executor A runtime environment interface permitting the
execution of shell commands (e.g., pytest ,
pip install ). It establishes a feedback loop
by capturing stdout /stderr for iterative de-
bugging and self-correction.
git_command Version control management interface. It ab-
stracts Git plumbing commands, enabling the
agent to manage repository state, branch for ex-
perimental features, and maintain a clean com-
mit history.
23