import ollama from "ollama";
import fs from "node:fs";

/*
╔══════════════════════════════════════════════════════════════════════════╗
║ CONTEXT & MEMORY — OVERVIEW                                               ║
╚══════════════════════════════════════════════════════════════════════════╝
| This file builds on the Agent Loop already implemented in grillman.ts
| (Agent Concept, State, Tools, Tool Selection, Idempotency, Guardrails —
| all of that is reused below, just with shorter comments, since it was
| already explained in detail there) and adds the CONTEXT & MEMORY layer:
|
| Context Engineering
|   → the general discipline of deciding WHAT goes into the LLM's context
|     window on each call, and WHY. Everything below is a specific
|     technique within that discipline.
|
| Context Selection
|   → given everything that EXISTS (the full state, the full history,
|     every customer's memory), choosing only what's RELEVANT right now.
|     We already did this with Tools (`getAvailableTools`); now we do the
|     same with information (`buildContext`).
|
| Compression
|   → the action history grows every iteration. Instead of sending
|     everything (expensive, and in grillman.ts we saw the small model
|     degrade with a long history) or nothing (we lose any sense of
|     trajectory — the problem we identified in grillman.ts), we
|     compress: a short summary of what's old + the most recent actions
|     in detail.
|
| Memory
|   → two levels. WORKING MEMORY is the `state` of ONE order (already
|     existed). LONG-TERM MEMORY is what we learn about a CUSTOMER across
|     multiple orders — it doesn't live in `state`, it lives separately
|     and survives past the end of the order.
|
| Persistence
|   → Long-term Memory is only useful if it survives past the end of the
|     Node process. Here that means: writing to disk (JSON) and reading
|     it back on the next start.
|
| Recovery
|   → what if the process dies MID-order (not at the end)? We persist a
|     checkpoint of the progress after every tool call, and on startup we
|     check whether there's a pending checkpoint to resume from — instead
|     of starting over and risking repeating side effects that already
|     happened for real.
*/

/*
╔══════════════════════════════════════════════════════════════════════════╗
║ STATE & TOOLS (imported from ./grillman_tools)                            ║
╚══════════════════════════════════════════════════════════════════════════╝
| The shape of the state, the Tool Schema, the Tool Metadata, and the
| executor with validation are the same ones used in grillman.ts — they
| live in ./grillman_tools.ts so they aren't duplicated between the two
| files. What changes here is just Tool Selection (getAvailableTools/
| currentPhase, below), which now shares the same phase reading used by
| Context Selection.
*/
import {
  type GrillmanState,
  type Doneness,
  type ToolResult,
  type IdempotencyStore,
  createInitialState,
  grillmanTools,
  executeGrillmanTool,
} from "./grillman_tools";
/*
| PHASE — a single function that answers "what's relevant RIGHT NOW?".
|
| In grillman.ts, Tool Selection and picking relevant state fields were
| two similar but separate pieces of logic. Here they're the SAME
| question asked twice — one filters possible ACTIONS, the other filters
| relevant INFORMATION — so they share this phase function.
*/
type Phase =
  | "fetching_meat"
  | "seasoning"
  | "waiting_for_grill"
  | "on_grill"
  | "ready_to_serve";

function currentPhase(state: GrillmanState): Phase {
  if (!state.meat.obtained) return "fetching_meat";
  if (!state.meat.seasoned) return "seasoning";
  if (!state.meat.onGrill && state.meat.currentDoneness === "raw")
    return "waiting_for_grill";
  if (state.meat.onGrill) return "on_grill";
  return "ready_to_serve";
}

function getAvailableTools(state: GrillmanState) {
  const byName = (names: string[]) =>
    grillmanTools.filter((tool) => names.includes(tool.function.name));

  /*
  | RELIABILITY FIX — `check_grill` as a low-risk "attractor".
  | In real testing (this file and grillman.ts), the small model
  | repeatedly preferred `check_grill` over an action that actually
  | advanced the order, whenever `check_grill` was available alongside
  | another option — likely because it's a side-effect-free Tool,
  | "always safe", and becomes the obvious fallback when the model is
  | uncertain.
  |
  | The fix isn't writing a better instruction asking it not to do that
  | — we already tried something like that and it didn't work. It's
  | removing the Tool from the list in phases where it doesn't change any
  | real decision: checking the temperature doesn't help choose between
  | "serve" or "grab the meat". It's only still offered in
  | `waiting_for_grill`, the one phase where checking the temperature
  | before acting is genuinely useful.
  */
  switch (currentPhase(state)) {
    case "fetching_meat":
      return byName(["get_meat"]);
    case "seasoning":
      return byName(["season_meat"]);
    case "waiting_for_grill":
      return byName(["put_on_grill", "check_grill"]);
    case "on_grill":
      return byName(["check_doneness", "flip_meat", "remove_from_grill"]);
    case "ready_to_serve":
      return byName(["serve_meat"]);
  }
}

/*
| Bumped up from 24: after the simulated cooling event, finishing the
| order can take a while if the model doesn't flip/check in a clean 1:1
| ratio (each wasted flip costs an iteration without advancing doneness).
| 24 was cutting it exactly at the point doneness finally matched the
| order, with no iterations left to call remove_from_grill/serve_meat.
*/
const MAX_ITERATIONS = 32;
const TIMEOUT_MS = 60_000;
const MAX_TOOL_CALLS = 32;

/*
╔══════════════════════════════════════════════════════════════════════════╗
║ MEMORY — two levels                                                       ║
╚══════════════════════════════════════════════════════════════════════════╝
| WORKING MEMORY already exists: it's the `GrillmanState` of ONE order. It
| is born in `createInitialState`, lives during the loop, and dies when
| the order ends (or survives only until the next restart, via checkpoint
| — see RECOVERY).
|
| LONG-TERM MEMORY is different: it's what remains about a CUSTOMER after
| their order has already ended — it spans multiple orders, multiple
| process restarts. It isn't part of `GrillmanState` because it doesn't
| belong to any specific order; it belongs to the relationship with the
| customer over time.
*/
interface CustomerMemory {
  totalOrdersCompleted: number;
  totalOrdersAborted: number;
  donenessLevelsOrdered: Doneness[];
  lastRelevantNote: string | null;
}

type LongTermMemory = Record<string, CustomerMemory>;

function emptyMemoryFor(): CustomerMemory {
  return {
    totalOrdersCompleted: 0,
    totalOrdersAborted: 0,
    donenessLevelsOrdered: [],
    lastRelevantNote: null,
  };
}

/*
| Called at the END of an order (success or abort via guardrail) to turn
| what happened into long-term memory about the customer. This is what
| connects one order to the next: without it, every order would be an
| island, and Grillman would never "learn" anything about anyone.
*/
function updateMemory(
  longTermMemory: LongTermMemory,
  state: GrillmanState,
): void {
  const customer = state.order.customer;
  const memory = longTermMemory[customer] ?? emptyMemoryFor();

  /*
  | WORDING MATTERS — this is Context Engineering too. An earlier version
  | of this note said "pay extra attention" and "consider more checking
  | cycles"; the small model reacted to that imperative/cautionary tone
  | by getting stuck repeating checking Tools (check_grill,
  | check_doneness) instead of advancing the order — a real side effect
  | of injecting "caution" text into the context of a model that already
  | tends to over-check. That's why the notes below are just descriptive
  | facts about what happened, with no behavioral instruction at all.
  */
  if (state.order.completed) {
    memory.totalOrdersCompleted += 1;
    memory.donenessLevelsOrdered.push(state.order.desiredDoneness);
    memory.lastRelevantNote = state.coolingEventTriggered
      ? "On the last order, the grill cooled down mid-preparation."
      : "The last order was completed without incident.";
  } else {
    memory.totalOrdersAborted += 1;
    memory.lastRelevantNote =
      "The last order didn't get completed within the iteration limit.";
  }

  longTermMemory[customer] = memory;
}

/*
╔══════════════════════════════════════════════════════════════════════════╗
║ PERSISTENCE — memory surviving past process exit                          ║
╚══════════════════════════════════════════════════════════════════════════╝
| Memory by itself only lives in this Node process's RAM — if the process
| ends (or restarts), `longTermMemory` would disappear with it.
| Persistence is the concrete mechanism that solves this: writing to
| something that survives (here, a JSON file on disk) and reading it back
| at the start of the next run.
|
| Run this script MORE THAN ONCE (`npx tsx grillman_with_context_and_
| memory.ts`) to see this for real: the memory written on the 1st run
| shows up already loaded at the start of the 2nd — in completely
| different Node processes, with nothing in common besides the file on
| disk.
*/
const MEMORY_PATH = "grillman-memory.json";

function loadLongTermMemory(): LongTermMemory {
  try {
    return JSON.parse(fs.readFileSync(MEMORY_PATH, "utf-8"));
  } catch {
    /* first run, or the file is corrupted/deleted: start from scratch */
    return {};
  }
}

function saveLongTermMemory(memory: LongTermMemory): void {
  fs.writeFileSync(MEMORY_PATH, JSON.stringify(memory, null, 2));
}

/*
╔══════════════════════════════════════════════════════════════════════════╗
║ COMPRESSION                                                               ║
╚══════════════════════════════════════════════════════════════════════════╝
| We keep the RAW history in full in `history` (useful for checkpointing,
| auditing, debugging) — it grows without bound, and that's fine, they're
| just short strings. The problem isn't storing it; it's what we SEND to
| the LLM every turn.
|
| In grillman.ts we fixed a reliability bug by throwing away EVERYTHING
| except the last observation — the model stopped degrading, but lost any
| sense of trajectory (the user themselves noticed this: "it doesn't
| remember if it already tried something"). Compression is the real
| middle ground:
|
| - the last N actions go in DETAILED (the model needs precision about
|   what just happened to decide the next step)
| - everything before that becomes a short SUMMARY (the model only needs
|   to know that it happened, not the exact details of each one)
|
| The size of what goes into the prompt stays O(1) relative to the number
| of iterations — it doesn't matter if the order already has 5 or 50
| actions, the context sent to the model always stays about the same
| size.
*/
const HISTORY_WINDOW = 3;

interface CompressedHistory {
  summary: string | null;
  recent: string[];
}

function compressHistory(history: string[], window: number): CompressedHistory {
  if (history.length <= window) {
    return { summary: null, recent: history };
  }

  const older = history.slice(0, history.length - window);
  const recent = history.slice(history.length - window);

  /*
  | The summary here is a count per action type — the simplest form of
  | compression there is (loses detail, keeps the fact "this happened, N
  | times"). A real system could ask another (cheaper) LLM to summarize
  | the old chunk in prose; here, a deterministic tally is already enough
  | for the didactic purpose.
  */
  const counts: Record<string, number> = {};
  for (const line of older) {
    const action = line.split(" → ")[0];
    counts[action] = (counts[action] ?? 0) + 1;
  }
  const parts = Object.entries(counts).map(([action, n]) => `${action} x${n}`);
  const summary = `Summary of ${older.length} earlier actions: ${parts.join(", ")}.`;

  return { summary, recent };
}

/*
╔══════════════════════════════════════════════════════════════════════════╗
║ CONTEXT SELECTION                                                         ║
╚══════════════════════════════════════════════════════════════════════════╝
| Builds the ONE message the LLM receives each turn (same pattern as
| grillman.ts: a fresh snapshot, not an accumulated conversation — that's
| what avoids the small model degrading over long contexts).
|
| The difference is what goes into that snapshot. It's not "the whole
| state, always" — it's an active selection:
|
| 1. Compressed trajectory (Compression, above) instead of nothing or
|    everything.
| 2. Long-term memory note for ONLY this customer — never every
|    customer's memory. This is already, in miniature, the same problem
|    RAG solves later in the roadmap: out of a large universe of
|    information, retrieving only the piece relevant to THIS question.
| 3. State fields relevant to the current PHASE, and only those — before
|    the meat goes on the grill, nobody needs to know the side or the
|    grill's temperature; after it comes off the grill, repeating
|    "seasoned: true" doesn't help at all, it's just noise.
*/
function buildContext(
  state: GrillmanState,
  history: string[],
  memoryNote: string | null,
): string {
  const { summary, recent } = compressHistory(history, HISTORY_WINDOW);

  const lines: string[] = [];
  lines.push(
    `Order: meat at "${state.order.desiredDoneness}" for ${state.order.customer}.`,
  );

  if (memoryNote) {
    lines.push(`Note from this customer's previous orders: ${memoryNote}`);
  }

  if (summary) {
    lines.push(summary);
  }
  lines.push(
    `Recent actions: ${recent.length > 0 ? recent.join(" | ") : "none yet"}.`,
  );

  switch (currentPhase(state)) {
    case "fetching_meat":
    case "seasoning":
      lines.push(
        `Meat obtained: ${state.meat.obtained}, seasoned: ${state.meat.seasoned}.`,
      );
      break;
    case "waiting_for_grill":
      lines.push(
        `Meat seasoned, still off the grill. Grill at ${state.grill.temperature}°C.`,
      );
      break;
    case "on_grill":
      lines.push(
        `On the grill, side ${state.meat.side}, current doneness "${state.meat.currentDoneness}". Grill at ${state.grill.temperature}°C.`,
      );
      /*
      | CONTEXT SELECTION reacting to state: the rule "stop checking once
      | doneness matches" already lives in SYSTEM_PROMPT, but a small
      | model tends to repeat the most recent pattern (check again)
      | instead of applying a conditional rule read a few lines above.
      | Instead of trusting only the static prompt, we compute the
      | condition HERE, at runtime, and only inject the warning when it's
      | true — an alert that shows up exactly on the turn it matters, and
      | never before that (this avoids the model seeing it and ignoring
      | it repeatedly until it actually applies).
      */
      if (state.meat.currentDoneness === state.order.desiredDoneness) {
        lines.push(
          `⚠️ The current doneness ALREADY MATCHES the order. Do NOT call check_doneness again — call remove_from_grill now.`,
        );
      } else {
        /*
        | RELIABILITY FIX — flip_meat also cools the grill (see
        | grillman_tools.ts) but does nothing for doneness by itself; only
        | check_doneness advances it. In real testing, the small model
        | sometimes called flip_meat two or three times in a row before
        | checking again, which only wastes heat (and iterations) for zero
        | cooking progress. This line makes that tradeoff explicit instead
        | of relying on the model to infer it from the Tool descriptions.
        */
        lines.push(
          "Only check_doneness advances the cooking — flipping without checking wastes heat for no benefit. Check after every flip.",
        );
      }
      break;
    case "ready_to_serve":
      /*
      | Even with only `serve_meat` offered (see getAvailableTools), the
      | small model has hallucinated `check_doneness` in free text at
      | this phase — it "thinks" there's one more step, from the generic
      | grilling script it learned. An explicit reminder that the meat is
      | ALREADY ready reduces that temptation.
      */
      lines.push(
        `Off the grill, current doneness "${state.meat.currentDoneness}" — already matches the order. No need to check again, just call serve_meat.`,
      );
      break;
  }

  lines.push("What's the next Tool?");
  return lines.join("\n");
}

const SYSTEM_PROMPT = `
You are a grill cook. Your goal is to fulfill ONE order from start to finish.

Every turn you receive a CONTEXT built for the current moment: a
compressed summary of what's already been done, the most recent actions
in detail, and (when there is one) a memory note about this customer.
Only the Tools relevant to the current phase are offered — choose among
them.

Rules:
- Never serve the meat (serve_meat) if the current doneness doesn't match the order.
- As soon as the current doneness EXACTLY matches the order, call
  remove_from_grill and then serve_meat — don't keep checking doneness.
- Once the order is complete, respond with a short sentence and don't
  call any more Tools.
`.trim();

/*
╔══════════════════════════════════════════════════════════════════════════╗
║ RECOVERY                                                                  ║
╚══════════════════════════════════════════════════════════════════════════╝
| Persistence (above) saves what already FINISHED. Recovery uses the same
| idea (writing to disk) for a different problem: what to do if the
| process dies MID-order — power outage, unhandled exception, the process
| got killed. Without this, restarting would mean starting from scratch,
| risking reapplying side effects that already happened FOR REAL in the
| real world (the meat has physically already been seasoned, whatever
| happens to the Node process).
|
| After every successful tool call, we save a checkpoint with `state` +
| `history` + the loop counters. When starting an order, we check whether
| there's a pending checkpoint for the SAME customer/order — if there is,
| we resume from there instead of calling `createInitialState`.
|
| Note that the Idempotency Key (from grillman.ts) doesn't need to
| survive the crash for this to work: a new process creates an empty
| `IdempotencyStore`, but the business-rule guards inside each `do*`
| handler (e.g. "meat is already seasoned") already block any improper
| repetition, because they read the recovered `state` — which already
| reflects reality. These are two independent layers of defense.
*/
interface Checkpoint {
  customer: string;
  desiredDoneness: Doneness;
  state: GrillmanState;
  history: string[];
  iteration: number;
  toolCallsUsed: number;
  startTime: number;
}

const CHECKPOINT_PATH = "checkpoint-grillman.json";

function saveCheckpoint(checkpoint: Checkpoint): void {
  fs.writeFileSync(CHECKPOINT_PATH, JSON.stringify(checkpoint, null, 2));
}

function loadCheckpoint(): Checkpoint | null {
  try {
    return JSON.parse(fs.readFileSync(CHECKPOINT_PATH, "utf-8"));
  } catch {
    return null;
  }
}

function clearCheckpoint(): void {
  try {
    fs.unlinkSync(CHECKPOINT_PATH);
  } catch {
    /* already gone — nothing to clear */
  }
}

/*
╔══════════════════════════════════════════════════════════════════════════╗
║ AGENT LOOP — putting it all together                                      ║
╚══════════════════════════════════════════════════════════════════════════╝
| `crashAfterToolCalls` exists just for the demo: instead of waiting for a
| real crash (e.g. `kill -9` mid-run), we force an exception after N tool
| calls. The RECOVERY mechanism that runs afterward is the same one that
| would run after a real process crash.
*/
async function runGrillmanAgent(
  customer: string,
  desiredDoneness: Doneness,
  longTermMemory: LongTermMemory,
  crashAfterToolCalls?: number,
): Promise<GrillmanState> {
  const idempotencyStore: IdempotencyStore = new Map();

  const existingCheckpoint = loadCheckpoint();
  const canResume =
    existingCheckpoint !== null &&
    existingCheckpoint.customer === customer &&
    existingCheckpoint.desiredDoneness === desiredDoneness;

  let state: GrillmanState;
  let history: string[];
  let iteration: number;
  let toolCallsUsed: number;
  let startTime: number;

  /*
  | STUCK DETECTOR — with temperature 0 (greedy decoding), a small model
  | can fall into an exact repetition trap: it hallucinates the same
  | malformed free-text "tool call" turn after turn, because the highest-
  | probability continuation never changes if nothing forces it to. In
  | testing, this happened right after `remove_from_grill` succeeded and
  | the phase moved on to `ready_to_serve` — the model kept trying to call
  | `remove_from_grill` again, in free text, 20+ times in a row, burning
  | the whole iteration budget without ever reaching `serve_meat`.
  |
  | The fix isn't a better prompt (we already inject an explicit reminder
  | for this exact phase) — it's breaking the determinism that's causing
  | the trap. We escalate temperature the more times this happens in a
  | row, so sampling has room to produce something other than the exact
  | same wrong answer. Normal decision-making stays at temperature 0.
  */
  let consecutiveFreeTextFailures = 0;

  if (canResume && existingCheckpoint) {
    console.log(
      `\n♻️  RECOVERY: found a checkpoint for ${customer} (iteration ${existingCheckpoint.iteration}, ${existingCheckpoint.toolCallsUsed} tool calls already done). Resuming instead of starting over.`,
    );
    state = existingCheckpoint.state;
    history = existingCheckpoint.history;
    iteration = existingCheckpoint.iteration;
    toolCallsUsed = existingCheckpoint.toolCallsUsed;
    startTime = existingCheckpoint.startTime;
  } else {
    state = createInitialState(customer, desiredDoneness);
    history = [];
    iteration = 0;
    toolCallsUsed = 0;
    startTime = Date.now();
  }

  /*
  | MEMORY feeding into CONTEXT SELECTION: we fetch only THIS customer's
  | memory (never the entire `longTermMemory`) to potentially inject into
  | the context.
  */
  const memoryNote = longTermMemory[customer]?.lastRelevantNote ?? null;

  while (true) {
    if (state.order.completed) {
      console.log(
        `\n✅ Order completed in ${iteration} iterations / ${toolCallsUsed} tool calls.`,
      );
      clearCheckpoint();
      break;
    }

    iteration++;

    if (iteration > MAX_ITERATIONS) {
      console.warn(
        `\n🛑 Max iterations (${MAX_ITERATIONS}) reached. Aborting order.`,
      );
      clearCheckpoint();
      break;
    }
    if (Date.now() - startTime > TIMEOUT_MS) {
      console.warn(`\n🛑 Timeout (${TIMEOUT_MS}ms) reached. Aborting order.`);
      clearCheckpoint();
      break;
    }
    if (toolCallsUsed >= MAX_TOOL_CALLS) {
      console.warn(
        `\n🛑 Cost limit (${MAX_TOOL_CALLS} tool calls) reached. Aborting order.`,
      );
      clearCheckpoint();
      break;
    }

    console.log(`\n--- Iteration ${iteration} ---`);

    const context = buildContext(state, history, memoryNote);
    if (iteration === 1) {
      console.log(`📋 Context sent to the model:\n${context}\n`);
    }

    const messages = [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: context },
    ];

    const temperature =
      consecutiveFreeTextFailures > 0
        ? Math.min(0.2 + consecutiveFreeTextFailures * 0.15, 0.9)
        : 0;

    const response = await ollama.chat({
      model: "llama3.2:latest",
      messages,
      tools: getAvailableTools(state),
      options: { temperature },
    });

    const toolCalls = response.message.tool_calls ?? [];

    if (toolCalls.length === 0) {
      if (state.order.completed) {
        console.log("Grillman:", response.message.content);
        break;
      }

      consecutiveFreeTextFailures++;
      console.warn(
        `  (model replied with free text instead of a tool call, ${consecutiveFreeTextFailures}x in a row, next temperature ${Math.min(
          0.2 + consecutiveFreeTextFailures * 0.15,
          0.9,
        ).toFixed(2)}):`,
        response.message.content,
      );
      history.push("(free text, reinforcing the instruction) → no tool call");
      saveCheckpoint({
        customer,
        desiredDoneness,
        state,
        history,
        iteration,
        toolCallsUsed,
        startTime,
      });
      continue;
    }

    consecutiveFreeTextFailures = 0;

    for (const [index, toolCall] of toolCalls.entries()) {
      if (toolCallsUsed >= MAX_TOOL_CALLS) break;
      toolCallsUsed++;

      const { name, arguments: args } = toolCall.function;
      const idempotencyKey = `${iteration}-${index}`;

      let result: ToolResult;
      try {
        result = await executeGrillmanTool(
          state,
          name,
          args,
          idempotencyStore,
          idempotencyKey,
        );
      } catch (error) {
        result = {
          ok: false,
          observation: "Failed to execute the Tool.",
          error: error instanceof Error ? error.message : "unknown_error",
        };
      }

      console.log(`  ${name} →`, result.observation);
      history.push(`${name} → ${result.observation}`);

      /*
      | RECOVERY in action: we persist progress after EVERY tool call,
      | not just at the end of the order. That's what makes the
      | checkpoint useful — it's never more than ONE action behind
      | reality.
      */
      saveCheckpoint({
        customer,
        desiredDoneness,
        state,
        history,
        iteration,
        toolCallsUsed,
        startTime,
      });

      if (
        crashAfterToolCalls !== undefined &&
        toolCallsUsed === crashAfterToolCalls
      ) {
        throw new Error(
          `💥 Simulating a process crash after ${toolCallsUsed} tool calls.`,
        );
      }
    }
  }

  console.log("\nFINAL STATE:");
  console.dir(state, { depth: null });
  return state;
}

async function main() {
  const longTermMemory = loadLongTermMemory();
  console.log("\n=== LONG-TERM MEMORY LOADED (read from disk) ===");
  console.dir(longTermMemory, { depth: null });

  const customer = "Ana";
  const desiredDoneness: Doneness = "medium";

  /*
  | We only simulate the crash when there ISN'T already a pending
  | checkpoint — otherwise every subsequent run would crash again before
  | finishing, and we'd never see the order complete or the long-term
  | memory get updated.
  */
  const crashAfterToolCalls = loadCheckpoint() === null ? 5 : undefined;

  let finalState: GrillmanState;
  try {
    finalState = await runGrillmanAgent(
      customer,
      desiredDoneness,
      longTermMemory,
      crashAfterToolCalls,
    );
  } catch (error) {
    console.warn(
      "\n🔥 Process 'crashed':",
      error instanceof Error ? error.message : error,
    );
    console.log(
      "The checkpoint stayed saved on disk. Run `npm run dev:grillman-memory` again to see RECOVERY.",
    );
    return;
  }

  updateMemory(longTermMemory, finalState);
  saveLongTermMemory(longTermMemory);

  console.log("\n=== LONG-TERM MEMORY UPDATED (written to disk) ===");
  console.dir(longTermMemory, { depth: null });
}

main();
