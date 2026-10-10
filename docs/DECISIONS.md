# Design decisions

Short log of choices and the reasoning behind them.

**D1: Explicit pipeline instead of fine-tuned Self-RAG.** The original Self-RAG fine-tunes a model with reflection
tokens. That needs GPUs and training data. Prompting an off-the-shelf LLM for each reflection step reproduces the idea
in a week and is easier to inspect and debug.

**D2: ChromaDB + sentence-transformers.** Free, local, persistent, zero infrastructure. `all-MiniLM-L6-v2` is small and
fast on CPU. Cosine distance with normalized vectors.

**D3: Batch grading (one call for all chunks).** Per-chunk grading costs k calls. Batching cuts cost and latency and
lets the grader compare passages. Tradeoff: slightly less isolated judgments.

**D4: Grader fails open, verifier fails closed.** If grading output is unparseable we keep the chunks (the verifier is
still the safety net). If verification output is unparseable we treat the answer as unverified, because accepting an
unchecked answer defeats the project's purpose.

**D5: Claim-level verification.** A single "is this grounded?" yes/no is too coarse and easy to satisfy. Per-claim
support gives a measurable ratio, a confidence score, and concrete feedback for the rewriter.

**D6: Bounded retries and explicit refusal.** Unbounded loops are a cost risk. Exhausting retries returns a refusal
rather than a best guess; users can trust that an answer was verified.

**D7: Word-based chunking (250 words, 40 overlap).** Simple, tokenizer-independent, good enough for papers. Improving it
(sentence/section-aware splitting) is future work.

**D8: Router is conservative.** It only bypasses retrieval for small talk; wrongly skipping retrieval on a factual
question is worse than one wasted search.

**D9: Dependency injection + lazy imports.** Enables fast, offline unit tests of the control flow.

**D10: Separate judge model for evaluation.** Reduces (but does not remove) self-preference bias.
