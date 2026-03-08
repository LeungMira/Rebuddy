# **AI Research Paper Verification & Summarization Tool**

A semantic research assistant that searches, verifies, and summarizes academic papers using structured verification pipelines and local LLM inference.

The system retrieves research papers from strictly defined academic sources, verifies their relevance through multiple semantic reasoning passes, and generates validated summaries formatted with ACM or APA citation styles.

The goal of this project is to reduce hallucinations and improve reliability when automatically processing research papers.

## Core Features
* Semantic Paper Search - The search module scans academic sources and retrieves papers based on semantic similarity rather than simple keyword matching. Key targets Include: Title, Abstract, and Publication Date.
* Multi-Stage Verification System - To reduce hallucinations or unstable summaries, the system performs triple summarization verification process.

Verification includes: 
* Three semantic evaluation passes
* If conflicting (polar) interpretations appear the system performs one final verification cycle
* If the final cycle fails, the paper is discarded

## Structured Paper Summarization
The summarization module extracts and analyzes the Abstract, Title and Key findings where the local LLM MISTRAL produces concise results.
