# **AI Research Paper Verification & Summarization Tool**

A semantic research assistant that searches, verifies, and summarizes academic papers using structured verification pipelines and local LLM inference.

The system retrieves research papers from strictly defined academic sources, verifies their relevance through multiple semantic reasoning passes, and generates validated summaries formatted with ACM or APA citation styles.

The goal of this project is to reduce hallucinations and improve reliability when automatically processing research papers.

## Core Features & Details
### Semantic Paper Search
The search module scans academic sources and retrieves papers based on semantic similarity rather than simple keyword matching. Key targets Include: Title, Abstract, and Publication Date.
### Multi-Stage Verification System 
To reduce hallucinations or unstable summaries, the system performs a triple phase verification process.

**Verification includes:**
* Three semantic evaluation passes
* If conflicting (polar) interpretations appear the system performs one final verification cycle
* If the final cycle fails, the paper is discarded

### Langgraph-CrossSearch Pipeline Documentation
The system utilizes LangGraph to enforce a deterministic execution path. By mapping the workflow as a state machine, we ensure strict adherence to safety rails and verification protocols. This architecture was specifically chosen to support recursive loops and iterative refinement without losing process integrity.

The CrossSearch pipeline is an object containtaining a 4-step LangGraph-based workflow designed to find, filter, validate, and output research papers from the CrossRef database. The system uses semantic similarity matching, metadata validation, and deduplication to ensure high-quality results.


### Search Engine 
The GUI captures user defined constraints including Target DOI/Library, Volume Limits (max 50), and Contextual Keywords, to instantiate a Search Engine Object. This object encapsulates the search configuration to optimize modularity and future changes. This design ensures Parameter Integrity and prevents API over utilization by enforcing hard coded limits at the object level

