# 🔥 Grillman

**An AI agent that grills meat — built to learn agent engineering hands-on, one real bug at a time.**

Grillman takes a single order (a piece of meat, a desired doneness) from raw to served, deciding every step itself: grab the meat, season it, put it on the grill, flip it, check it, serve it. It's a toy domain on purpose — small enough to fully understand, real enough to expose the same failure modes production agents run into.

## Why this exists

The market increasingly expects engineers to actually understand how AI agents work, not just call an API and hope. Rather than stopping at tutorials, this repo works through the full roadmap in [`studies.md`](./studies.md) hands-on — implementing each concept, running it against a real local LLM, and documenting whatever broke along the way.

No mocked model, no scripted happy path. Grillman runs against a real (small, imperfect) model via [Ollama](https://ollama.com), which means it hallucinates, gets confused, and needs real guardrails — exactly the conditions that force you to actually learn the concepts instead of just reading about them.

## What's in here

| File | Roadmap phase | What it covers |
|---|---|---|
| [`grillman_tools.ts`](./grillman_tools.ts) | **Tools** | The world state, Tool Schema (what the LLM sees), Tool Metadata (side effects, idempotency, retry policy — what only the runtime sees), and a validated executor that never trusts the model's output blindly. |
| [`grillman.ts`](./grillman.ts) | **Agent Concept, Agent Loop, Tool Selection, Guardrails** | The observe → decide → act → observe loop, Tool Selection to narrow the model's choices, and hard limits (max iterations, timeout, cost) so a reasoning bug can't run forever. |
| [`grillman_with_context_and_memory.ts`](./grillman_with_context_and_memory.ts) | **Context Engineering, Compression, Memory, Persistence, Recovery** | Builds a fresh, selected context every turn instead of an ever-growing transcript, remembers customers across orders, persists that memory to disk, and recovers cleanly if the process dies mid-order. |

Everything reuses the same tools and the same "world" — what changes across files is how much the loop is allowed to reason about and remember, layer by layer.

## The roadmap

[`studies.md`](./studies.md) is the full plan this project works through, phase by phase: LLM Fundamentals → Structured Generation → Tools → Agents → Context & Memory → RAG → Production → Security → Harness → Advanced.

This repo currently covers everything through **Context & Memory**. RAG is next.

## Real bugs found along the way

The point of building instead of reading is that a toy problem still breaks in real ways:

- **A hallucinated tool.** The model repeatedly tried to call `temper` — a tool that never existed — instead of `season_meat`. Fixed by treating every tool call as untrusted input, validated at the boundary, never executed on trust.
- **An addiction to the "safe" action.** Given too many valid-looking tools at once, the model kept calling the harmless, side-effect-free `check_grill` instead of finishing the order. The fix wasn't a better prompt — it was narrowing the model's choices to only what's actually possible in the current state.
- **A memory note that changed behavior through tone alone.** A long-term memory note that said "pay extra attention" (no new facts, just caution) made the model get stuck over-checking doneness. Rewriting it as a neutral, descriptive fact fixed the loop — proof that everything injected into a model's context functions as an instruction, whether you meant it to or not.

## Running it

Requires [Ollama](https://ollama.com) running locally with a pulled model:

```bash
ollama pull llama3.2
```

Then:

```bash
npm install

npm run dev:grillman          # Agent Loop + Tools + Tool Selection
npm run dev:grillman-memory   # + Context Engineering + Memory + Persistence + Recovery
```

`dev:grillman-memory` writes long-term memory to `grillman-memory.json` and a live checkpoint to `checkpoint-grillman.json` (both gitignored) — run it more than once to see memory persist across restarts, and let it "crash" mid-order to see recovery kick in.
