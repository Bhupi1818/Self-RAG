# Pipeline: step by step

Implemented in `src/pipeline.py` (`SelfRAGPipeline.answer`).

## Step 1: Router
- Input: question. Output: `needs_retrieval` (bool).
- `false` only for greetings/thanks/small talk. Any parse error returns `true`.

## Step 2: Retrieve
- Embed the current query (`all-MiniLM-L6-v2`, normalized) and fetch `TOP_K` chunks by cosine similarity.
- First attempt uses the user's question; later attempts use the rewritten query.

## Step 3: Grade
- One LLM call receives the question and all chunks, returns `{"grades": [{"id", "relevant", "reason"}]}`.
- Topical similarity is not enough; a chunk must contain facts that help answer.
- If fewer than `MIN_RELEVANT` chunks survive, skip generation and go to Step 6 (saves a generation call).

## Step 4: Generate
- Prompt contains only relevant chunks, numbered `[1]..[n]`, with source and page.
- The model must cite, use only the context, and otherwise output the exact refusal sentence.
- If it refuses, treat as a failed attempt (no verifier call).

## Step 5: Verify
- LLM splits the answer into atomic claims and marks each supported / unsupported by the context.
- `supported_ratio = supported / total claims`; also `answers_question`.
- Accept only if `supported_ratio >= SUPPORT_THRESHOLD` (default 0.8) and `answers_question`.
- Parse failure counts as "not verified" (fail-closed).

## Step 6: Rewrite and retry
- The rewriter gets the original question, previous query, and the failure reason (e.g. "only 1 relevant passage" or the
  unsupported claims) and returns a better query. Loop back to Step 2.
- At most `MAX_RETRIES` rewrites (default 2, so up to 3 attempts).

## Step 7: Refusal
- If every attempt fails: return "I don't have enough information in the provided documents to answer this." with
  `refused=true`, `confidence=0`.

## Output
`answer`, `sources` (id, source, page, score, snippet), `confidence` (= supported ratio), `attempts`, `refused`,
`trace`, `llm_calls`, `latency_s`.

## Cost per query (typical)
Happy path: router + grade + generate + verify = 4 LLM calls. Each retry adds rewrite + grade (+ generate + verify).
Baseline RAG uses 1 call. This tradeoff is reported in the evaluation.
