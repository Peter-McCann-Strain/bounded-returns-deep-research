# Future Work: Agentic Search Patterns from Chroma Context-1

**Status**: Planned — to be implemented after current evaluation campaign completes
**Date**: 2026-03-30
**Reference**: [Chroma Context-1 Research](https://www.trychroma.com/research/context-1), [HuggingFace model](https://huggingface.co/chromadb/context-1)

## Background

Chroma released Context-1 (March 2026), a 20B parameter agentic search model trained with SFT + RL (CISPO) on gpt-oss-20b. It pushes the Pareto frontier of agentic search — achieving frontier-model retrieval quality at 10x faster inference and 25x lower cost. Several of its architectural patterns directly address weaknesses we identified in our P0-P10 evaluation.

### Key Finding From Our Evaluation

P1 Iterative RAG leads (0.763) because of:
- **1-hop information distance** between evidence and report (vs 2-6 hops for P2-P4)
- **Iterative reflection loop** that closes citation/recall gaps
- **High source volume** (53+ real-URL references vs 5-20 for others)

But P1 also has weaknesses: catastrophic failures when initial search goes wrong (bimodal distribution), no context pruning (noise accumulates), and no separation between search and generation concerns. Context-1's patterns address all three.

---

## Pattern 1: Self-Editing Context Pruning

### What Context-1 Does

The model has a `prune_chunks` tool and a three-tier pressure system for managing its context window:

1. **Continuous visibility**: Token usage appended after each turn ("14,203/32,768 tokens")
2. **Soft threshold**: At ~90% capacity, harness injects a message suggesting pruning
3. **Hard cutoff**: Beyond 31-32k tokens, all tools except pruning are rejected

Pruning accuracy improved from 0.824 (base model) to 0.941 after RL training. Pruned chunks are removed from the model's view but preserved for reward computation during training.

### How to Apply to Our Patterns

**Target**: P1 Iterative RAG (highest priority), P6 Reactive Interleaved, P7 Graph Decomposition

**Implementation approach**:

1. After each search iteration in P1's reflection loop, add a **pruning step** before report regeneration:
   ```
   extractions = await source_extractor.extract_batch(docs, query)
   # NEW: Prune irrelevant extractions before passing to generator
   pruned_extractions = await prune_extractions(llm, extractions, query, max_tokens=budget)
   report_md = await generator.generate(query, pruned_extractions, ...)
   ```

2. The pruning prompt asks the LLM to score each extraction's relevance to the query and discard those below a threshold, or to select the top-K most relevant.

3. Add token budget visibility to the generator prompt: "You have {N} source extractions using approximately {T} tokens. Focus on the most relevant sources."

**Expected impact**: Reduces P1's catastrophic failures (where irrelevant sources accumulate and drown out signal). Should improve factual_accuracy and information_recall on queries where initial search returns mixed-quality results.

**Ablation design**: Run P1 with and without pruning on the same 90 queries. Compare overall_score distributions, especially the lower tail (scores < 0.4).

---

## Pattern 2: Search-Generation Separation (Tiered Architecture)

### What Context-1 Does

Context-1 functions as a **retrieval subagent** (Tier 1) that returns a ranked document set to a separate downstream reasoning model (Tier 2). This separation:

- Prevents confounding (retrieval quality measured independently of generation quality)
- Allows a smaller specialised model for search + frontier model for synthesis
- Eliminates context noise that causes the generation model to hallucinate

Their recommended production architecture:
- **Tier 1 (Search)**: Context-1 (20B) optimised for recall, precision, multi-hop reasoning
- **Tier 2 (Generation)**: Frontier LLM optimised for reasoning, instruction-following, response quality
- **Tier 3 (Infrastructure)**: Document store, vector database, BM25 index

### How to Apply to Our Patterns

**Target**: New pattern P11 (or P1 variant)

**Implementation approach — P11 "Tiered Search-Synthesis"**:

1. **Search agent** (could be GPT-4o-mini or a local model): Runs iterative multi-hop search with query decomposition, source retrieval, and context pruning. Outputs a curated, ranked evidence package with relevance scores.

2. **Synthesis agent** (GPT-4o): Receives only the pruned evidence package. Generates the research report with full citation grounding. Never touches the search infrastructure directly.

3. The search agent's output format:
   ```json
   {
     "query": "...",
     "evidence": [
       {"rank": 1, "url": "...", "title": "...", "summary": "...", "relevance": 0.95, "key_findings": [...]},
       ...
     ],
     "search_trace": ["subquery1", "subquery2", ...],
     "pruned_count": 15,
     "total_retrieved": 68
   }
   ```

4. This cleanly separates the two optimisation targets: the search agent is judged on recall/precision, the synthesis agent on report quality.

**Expected impact**: Combines P1's retrieval strength with P4's synthesis depth. The synthesis agent sees only high-quality, pre-ranked evidence — no "Web Search Synthesis" placeholders, no irrelevant noise.

**Ablation design**: Compare P11 vs P1 (same search budget) and P11 vs P4 (same synthesis model). The key question: does explicit separation improve citation_quality and attribution_quality without sacrificing analytical_depth?

---

## Pattern 3: Hybrid BM25 + Dense Retrieval with RRF

### What Context-1 Does

Context-1's `search_corpus` tool uses hybrid retrieval: BM25 (keyword) + dense (semantic) with Reciprocal Rank Fusion (RRF), retrieving 50 candidates then reranking. It also has a separate `grep_corpus` tool for exact regex matching.

### How to Apply to Our Patterns

**Target**: All patterns (infrastructure-level improvement)

**Implementation approach**:

1. After web search retrieves documents, add a **local re-ranking step** before source extraction:
   - Compute BM25 scores against the original query + sub-queries
   - Compute dense similarity using a lightweight embedding model
   - Combine with RRF: `score = 1/(k + rank_bm25) + 1/(k + rank_dense)` where k=60
   - Pass top-N reranked documents to source extraction

2. For the `SourceExtractor`, prioritise documents by combined RRF score rather than retrieval order.

3. Add a `grep_corpus` equivalent: after initial retrieval, search within retrieved document text for exact query terms, names, dates. Boost documents with exact matches.

**Expected impact**: Moderate improvement across all patterns. Should help on queries where semantic search retrieves topically related but not specifically relevant documents.

**Dependencies**: Requires a lightweight embedding model (e.g., `all-MiniLM-L6-v2` or similar). Minimal compute overhead.

---

## Pattern 4: Token Budget Visibility in Prompts

### What Context-1 Does

After every tool call, Context-1's harness appends token usage: "14,203/32,768 tokens". This continuous visibility helps the model reason about when to stop searching and when to prune.

### How to Apply to Our Patterns

**Target**: P6 Reactive Interleaved, P7 Graph Decomposition (model-driven control flow patterns)

**Implementation approach**:

1. In patterns where the model decides when to search vs. when to stop (P6, P7), include context budget information in each iteration's prompt:
   ```
   [Budget: {n_sources} sources retrieved, ~{token_estimate} tokens of evidence,
   {iterations_remaining} search iterations remaining, ${cost_so_far:.3f} of ${budget:.2f} spent]
   ```

2. This gives the model information to make better stop/continue decisions without hard-coding iteration limits.

**Expected impact**: Low-effort improvement for P6/P7. May reduce cases where these patterns over-search (wasting budget) or under-search (stopping too early).

---

## Pattern 5: RL-Trained Search Agent (Longer-Term)

### What Context-1 Does

Context-1 uses CISPO (Clipped Importance-Sampled Policy Optimization) — a GRPO variant — with a multi-component reward:

- **F1 score** with 16:1 recall-to-precision weighting (encourages broad exploration early)
- **Trajectory recall bonus** — credits finding relevant docs even if later pruned
- **Penalties**: consecutive prune penalty, linear turn penalty
- **Reward annealing**: 16:1 → 4:1 recall:precision ratio over training (explore broadly, then learn selectivity)

Training used 128 queries/step x 8 rollouts = 1,024 trajectories/step, converging at ~230 steps across 5 epochs.

### How to Apply to Our Patterns

**Target**: Long-term — potential P12 or replacement for P10

**Implementation approach**:

1. Use Context-1's [open-source data generation pipeline](https://github.com/chroma-core/context-1-data-gen) to create domain-specific search training data for our evaluation queries.

2. Fine-tune a smaller model (Qwen2.5-7B or similar) with GRPO on search trajectories where the reward is based on downstream report quality (judge scores) rather than just retrieval recall.

3. The key insight from Context-1: **train the model within the exact tool harness it will use in production**. The agent harness (system prompt, tool definitions, loop structure) becomes part of learned behaviour, not optional scaffolding.

**Expected impact**: High but requires significant compute investment. Context-1 shows RL-trained search agents at 20B can match frontier models — our P10 at 7B was likely too small and insufficiently trained.

**Prerequisites**: GPU budget for RL training (RTX 5080 may be sufficient for 7B LoRA), synthetic data pipeline, reward model based on judge scores.

---

## Pattern 6: Synthetic Evaluation Data Generation

### What Context-1 Does

Four-stage pipeline per domain:
1. **Gather docs**: Agent explores corpus collecting unique facts
2. **Generate tasks**: Creates obfuscated clues, question, answer
3. **Verify**: Extraction-based verification (LLM extracts quotes, normalised matching confirms evidence, 84-93% alignment with human labels)
4. **Add distractors**: Documents satisfying some criteria but yielding different answers
5. **Chain** (optional): Bridge answers across tasks to control hop count

### How to Apply to Our Patterns

**Target**: Evaluation framework improvement

**Implementation approach**:

1. Use Context-1's verification methodology to create harder evaluation queries with known ground-truth retrieval chains. Our current DRACO and DeepSearchQA benchmarks test answer quality but not retrieval quality.

2. Generate multi-hop evaluation tasks where we know exactly which documents must be retrieved (and in what order) to answer correctly. This would let us measure retrieval recall independently of report quality.

3. Add distractor documents to evaluation queries — documents that are topically similar but don't contain the needed information. This tests whether patterns can distinguish relevant from merely related sources.

**Expected impact**: Better diagnostic evaluation. Currently we measure report quality but can't distinguish "good search, bad synthesis" from "bad search, good synthesis." Source-level ground truth would decompose the problem.

---

## Implementation Priority

| Priority | Pattern | Effort | Expected Impact | Depends On |
|----------|---------|--------|-----------------|------------|
| 1 | Context pruning for P1 | 2-3 days | High — fixes P1 failure mode | Nothing |
| 2 | Token budget visibility | 1 day | Moderate — helps P6/P7 | Nothing |
| 3 | BM25+dense reranking | 2-3 days | Moderate — all patterns | Embedding model |
| 4 | P11 Tiered Search-Synthesis | 1 week | High — new pattern | Pruning (#1) |
| 5 | Synthetic multi-hop eval data | 1 week | High — better diagnostics | Nothing |
| 6 | RL-trained search agent | 2-4 weeks | Very high but speculative | GPU budget, data pipeline |

---

## References

- [Chroma Context-1 Research Paper](https://www.trychroma.com/research/context-1)
- [Chroma Context-1 Product Page](https://www.trychroma.com/products/agent)
- [HuggingFace: chromadb/context-1](https://huggingface.co/chromadb/context-1)
- [GitHub: context-1-data-gen](https://github.com/chroma-core/context-1-data-gen)
- [Blog: Self-Editing Search Agent](https://atalupadhyay.wordpress.com/2026/03/27/chroma-context-1-the-self-editing-search-agent-that-changes-how-we-build-rag-systems/)
- [MarkTechPost: Chroma Releases Context-1](https://www.marktechpost.com/2026/03/29/chroma-releases-context-1-a-20b-agentic-search-model-for-multi-hop-retrieval-context-management-and-scalable-synthetic-task-generation/)
- [How Kimi, Cursor, and Chroma Train Agentic Models with RL](https://www.philschmid.de/kimi-composer-context)
