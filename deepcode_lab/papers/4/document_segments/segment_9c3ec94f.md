# Result

**Content Type:** experiment
**Keywords:** advantage, generalization, than, improvement, best, configuration, capability, procedural, substantially, unified

Results
This appendix reports quantitative results that complement the main text and provide a more systematic
evaluation of DeepCode’s overall capability and stability on research code reproduction tasks. Table 2
first compares, under a unified evaluation protocol, a range of general-purpose code execution agents
(including both BasicAgent and IterativeAgent configurations), existing specialized reproduction
systems such as PaperCoder, and human experts on the same benchmark. DeepCode achieves
an average reproduction score of 73.5±2.8 on the full benchmark, substantially outperforming
PaperCoder ( 51.1±1.4 ) as well as all configurations derived from commercial models. On the
3-paper subset, DeepCode attains an average score of 75.9±4.5 , exceeding the human “Best@3”
score of 72.4, indicating that, on representative deep learning papers, the system delivers reproduction
quality comparable to or better than that of strong human practitioners.
Table 1 further selects a 5-paper subset (fre, rice, bam, pinn, mech-u) for a head-to-head comparison
against several widely used commercial code assistants (Codex, Claude Code, Cursor, etc.). Across all
papers, DeepCode achieves the highest reproduction score, with an average of 0.8482, corresponding
to an absolute improvement of more than 0.26 over the strongest competing system. The advantage
is consistent across all individual papers, suggesting that the gains arise from architectural and
procedural design choices rather than from favorable alignment with a narrow subset of tasks.
Finally, Table 3 provides per-paper details for the Claude 4.5 Sonnet–based configuration, includ-
ing three independent runs, their mean and standard error, as well as the associated average cost.
Across a diverse set of targets—such as FRE, PINN, MECHANISTIC-UNDERSTANDING, and
SEQUENTIAL-NEURAL-SCORE-ESTIMATION—DeepCode’s reproduction scores typically lie
in the 0.7–0.9 range with relatively small standard errors, while the distribution of average cost
across papers remains tight. This indicates strong cross-task generalization, stable behavior across
repeated runs, and reasonable resource usage. Taken together, these appendix results reinforce the
main