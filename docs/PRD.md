# Product Requirements: Self-Reflective RAG

## Problem
Standard RAG retrieves top-k chunks and generates an answer regardless of chunk quality. This causes (1) answers built on
irrelevant context, (2) hallucinated details, and (3) confident answers to questions the corpus cannot answer.

## Goal
Build a RAG system that detects and corrects these failures at inference time, and prove the improvement with a measured
comparison against plain RAG.

## Users & stories
- *Student/researcher*: "Ask questions over a set of papers and see exactly which passages support the answer."
- *Developer evaluating RAG*: "Compare baseline vs self-reflective answers on the same questions."
- *Reviewer/recruiter*: "Run it in 5 minutes and read a clear architecture and results table."

## Functional requirements
1. Ingest PDF/TXT/MD documents into a persistent vector store.
2. Answer questions with inline citations and source passages.
3. Route trivial/small-talk queries away from retrieval.
4. Grade retrieved passages; discard irrelevant ones.
5. Verify the answer claim-by-claim against the context.
6. Rewrite the query and retry (bounded) when retrieval or verification fails.
7. Refuse gracefully when no grounded answer exists.
8. Expose a REST API and a UI showing the full reasoning trace.
9. Provide a baseline mode and an evaluation harness.

## Non-goals
Multi-user auth, streaming, fine-tuning, multi-modal inputs, production-scale indexing.

## Success metrics
- Higher faithfulness than baseline on the same test set.
- Correct refusal on unanswerable questions (refusal accuracy) without large false-refusal rate.
- Cost transparency: average LLM calls and latency per query reported.

## Constraints
One-week build, free/local embeddings, small hosted LLM or local Ollama model, runs on a laptop.
