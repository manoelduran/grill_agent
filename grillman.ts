import ollama from "ollama";

/*
╔══════════════════════════════════════════════════════════════════════════╗
║ AGENT CONCEPT                                                             ║
╚══════════════════════════════════════════════════════════════════════════╝
| An Agent is not "a smart function". It's a LOOP that:
|
|   observes the state of the world
|   ↓
|   decides the next action (using an LLM)
|   ↓
|   acts through a Tool
|   ↓
|   observes the result of that action
|   ↓
|   decides again...
|
| until the order is done OR some guardrail (a limit) stops it.
|
| Customer → Order → Agent (Grillman) → Tools → Grill state
|
| The whole "Agent", in this file, is Grillman: the LLM + the rules for
| when it can act, how many times, for how long, and with which Tools.
*/

/*
╔══════════════════════════════════════════════════════════════════════════╗
║ STATE & TOOLS (imported from ./grillman_tools)                            ║
╚══════════════════════════════════════════════════════════════════════════╝
| The shape of the state, the Tool Schema, the Tool Metadata, and the
| executor with validation (idempotency, retry policy, permission
| boundary) live in ./grillman_tools.ts — they're the same for this file
| and for grillman_with_context_and_memory.ts, so they moved to one place.
*/
import {
  type GrillmanState,
  type Doneness,
  type ToolResult,
  type IdempotencyStore,
  createInitialState,
  grillmanTools,
  toolMetadata,
  executeGrillmanTool,
} from "./grillman_tools";

/*
╔══════════════════════════════════════════════════════════════════════════╗
| TOOL SELECTION — narrowing the decision space                             ║
╚══════════════════════════════════════════════════════════════════════════╝
| Offering all 8 Tools at once, always, is what makes a small model get
| lost: in real testing, llama3.2:3b repeatedly tried to call a made-up
| Tool ("temper") instead of `season_meat`, because it had too much
| ambiguity to choose from among 8 similar-looking options every turn.
|
| The fix isn't "trust the LLM more" — it's giving it FEWER valid choices
| at a time. Each state of the meat only makes a subset of actions
| physically possible (you can't flip meat that isn't on the grill); so
| we only expose those. This is runtime-guided Tool Selection, and it
| anticipates what we'll see later in Context Engineering / Context
| Selection: fewer irrelevant options in the context = a more reliable
| decision.
*/
function getAvailableTools(state: GrillmanState) {
  const byName = (names: string[]) =>
    grillmanTools.filter((tool) => names.includes(tool.function.name));

  /*
  | Order matters: small models tend to favor the first option in the
  | list when they're uncertain. That's why the expected action for the
  | current state always comes first, and `check_grill` (optional,
  | informative) comes last.
  */
  if (!state.meat.obtained) {
    return byName(["get_meat", "check_grill"]);
  }
  if (!state.meat.seasoned) {
    return byName(["season_meat", "check_grill"]);
  }
  if (!state.meat.onGrill && state.meat.currentDoneness === "raw") {
    return byName(["put_on_grill", "check_grill"]);
  }
  if (state.meat.onGrill) {
    return byName([
      "check_doneness",
      "flip_meat",
      "remove_from_grill",
      "check_grill",
    ]);
  }
  return byName(["serve_meat", "check_grill"]);
}

/*
╔══════════════════════════════════════════════════════════════════════════╗
| GUARDRAILS — MAX ITERATIONS / TIMEOUT / COST LIMITS                       ║
╚══════════════════════════════════════════════════════════════════════════╝
| Without these limits, an LLM reasoning bug (e.g. calling check_doneness
| repeatedly without ever flipping the meat) would make the Agent run
| forever.
|
| MAX_ITERATIONS → stops Grillman from checking doneness 500 times.
| TIMEOUT_MS     → the order can't keep running indefinitely.
| MAX_TOOL_CALLS → a simple proxy for "cost limit" (in production, this
|                  would be tokens consumed / model calls; here we count
|                  Tool calls for simplicity).
*/
const MAX_ITERATIONS = 24;
const TIMEOUT_MS = 45_000;
const MAX_TOOL_CALLS = 24;

/*
╔══════════════════════════════════════════════════════════════════════════╗
| AGENT LOOP                                                                ║
╚══════════════════════════════════════════════════════════════════════════╝
| observe state → decide (LLM + Tools) → act (run Tool) →
| observe result → decide again.
|
| PLANNING happens implicitly: the system prompt describes the natural
| plan ("grab, season, grill, flip, check, remove, serve"), but the LLM
| itself decides the order call by call, based on what it observes — it's
| not a fixed plan executed blindly.
*/
const SYSTEM_PROMPT = `
You are a grill cook. Your goal is to fulfill ONE order from start to finish.

Every turn you receive a summary of the CURRENT STATE (not the full
conversation history) and must call exactly one Tool: the next action
that's physically possible from that state. Only the Tools relevant to
the current state are offered to you — choose among them.

Rules:
- Never serve the meat (serve_meat) if the current doneness doesn't match the order.
- As soon as the current doneness EXACTLY matches the order, call
  remove_from_grill and then serve_meat — don't keep checking doneness,
  that only overcooks the meat.
- Once the order is complete, respond with a short sentence and don't
  call any more Tools.
`.trim();

/*
| Describes the current state in short text for the LLM. This is the
| ONLY source of "memory" the model gets each turn — not the conversation
| transcript. The real state (source of truth) keeps living in `state`,
| outside the LLM; the model only observes a fresh read of it, always
| O(1) in size, never growing with the number of iterations.
*/
function describeState(state: GrillmanState, lastObservation: string): string {
  return `
Order: meat at "${state.order.desiredDoneness}" for ${state.order.customer}.
Last observation: ${lastObservation}
Meat state: obtained=${state.meat.obtained}, seasoned=${state.meat.seasoned}, on grill=${state.meat.onGrill}, side=${state.meat.side ?? "-"}, current doneness="${state.meat.currentDoneness}".
Grill: ${state.grill.temperature}°C, on=${state.grill.on}.
What's the next Tool?
`.trim();
}

async function runGrillmanAgent(customer: string, desiredDoneness: Doneness) {
  const state = createInitialState(customer, desiredDoneness);
  const idempotencyStore: IdempotencyStore = new Map();
  const startTime = Date.now();
  let iteration = 0;
  let toolCallsUsed = 0;
  let lastObservation = "Order just came in, nothing has been done yet.";

  while (true) {
    /*
    |------------------------------------------------------------------
    | TERMINATION — success
    |------------------------------------------------------------------
    */
    if (state.order.completed) {
      console.log(
        `\n✅ Order completed in ${iteration} iterations / ${toolCallsUsed} tool calls.`,
      );
      break;
    }

    iteration++;

    /*
    |------------------------------------------------------------------
    | TERMINATION — safety guardrails (not success, it's an abort)
    |------------------------------------------------------------------
    */
    if (iteration > MAX_ITERATIONS) {
      console.warn(
        `\n🛑 Max iterations (${MAX_ITERATIONS}) reached. Aborting order.`,
      );
      break;
    }
    if (Date.now() - startTime > TIMEOUT_MS) {
      console.warn(`\n🛑 Timeout (${TIMEOUT_MS}ms) reached. Aborting order.`);
      break;
    }
    if (toolCallsUsed >= MAX_TOOL_CALLS) {
      console.warn(
        `\n🛑 Cost limit (${MAX_TOOL_CALLS} tool calls) reached. Aborting order.`,
      );
      break;
    }

    console.log(`\n--- Iteration ${iteration} ---`);

    /*
    |------------------------------------------------------------------
    | DECIDE — the LLM observes a fresh snapshot of the state (not the
    | accumulated history) and decides the next action.
    |------------------------------------------------------------------
    */
    const messages = [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: describeState(state, lastObservation) },
    ];

    const response = await ollama.chat({
      model: "llama3.2:latest",
      messages,
      tools: getAvailableTools(state),
      options: { temperature: 0 },
    });

    const toolCalls = response.message.tool_calls ?? [];

    if (toolCalls.length === 0) {
      if (state.order.completed) {
        console.log("Grillman:", response.message.content);
        break;
      }

      /*
      | RELIABILITY — small models sometimes "narrate" a tool call in
      | free text instead of using the real tool-calling mechanism. This
      | isn't success (the order isn't complete), so we don't treat it as
      | termination — we log the failure as an observation and let the
      | loop try again, still protected by the guardrails.
      */
      console.warn(
        "  (model replied with free text instead of a tool call):",
        response.message.content,
      );
      lastObservation =
        "You replied with free text last turn. Call a real Tool now.";
      continue;
    }

    /*
    |------------------------------------------------------------------
    | ACT + OBSERVE — sequential on purpose.
    |------------------------------------------------------------------
    | Unlike get_weather (parallel, because the calls were independent),
    | here the actions mutate a SINGLE shared state (the same meat, the
    | same grill). Running in parallel would create race conditions (e.g.
    | put_on_grill and season_meat deciding based on the same stale
    | state). Sequential side effects on shared state must be sequential.
    */
    for (const [index, toolCall] of toolCalls.entries()) {
      if (toolCallsUsed >= MAX_TOOL_CALLS) break;
      toolCallsUsed++;

      const { name, arguments: args } = toolCall.function;

      /*
      | Every tool_call the LLM returns here is a NEW DECISION (Ollama
      | doesn't give us a tool_call id, unlike APIs like OpenAI's or
      | Anthropic's). That's why we generate a fresh key per decision —
      | there's no automatic retry of side effects in this loop, so never
      | reusing a key here is the correct behavior. A future extension
      | with real retries (e.g. a network error) would reuse the SAME key
      | when repeating the same logical attempt, instead of generating a
      | new one.
      */
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
      lastObservation = `${name} → ${result.observation}`;
    }
  }

  console.log("\nFINAL STATE:");
  console.dir(state, { depth: null });
  return state;
}

/*
╔══════════════════════════════════════════════════════════════════════════╗
| TESTING IDEMPOTENCY                                                       ║
╚══════════════════════════════════════════════════════════════════════════╝
| Outside the agent loop, we directly test the idempotency guarantees the
| metadata promises — without depending on the LLM to call things in the
| right order.
*/
async function testIdempotency() {
  console.log("\n=== IDEMPOTENCY TEST ===");

  const store: IdempotencyStore = new Map();

  /*
  | check_grill (idempotent, read-only): calling it twice on the same
  | state should give the exact same observation. Idempotent Tools don't
  | even go through the cache — that's why the keys below can be
  | different and irrelevant, and the result still matches.
  */
  const state = createInitialState("Test", "medium");
  const r1 = await executeGrillmanTool(
    state,
    "check_grill",
    {},
    store,
    "irrelevant-1",
  );
  const r2 = await executeGrillmanTool(
    state,
    "check_grill",
    {},
    store,
    "irrelevant-2",
  );
  console.log("check_grill 1st call:", r1.observation);
  console.log("check_grill 2nd call:", r2.observation);
  console.log(
    "Idempotent? ",
    JSON.stringify(r1) === JSON.stringify(r2) ? "YES" : "NO",
  );

  /*
  | BUSINESS-RULE GUARD ≠ IDEMPOTENCY KEY
  | season_meat called twice with DIFFERENT keys (two new decisions, not
  | a retry of the same attempt): the 2nd call is blocked by the domain
  | rule inside doSeasonMeat. Notice the results are DIFFERENT from each
  | other (ok:true then ok:false) — that proves this guard isn't
  | recognizing "I already processed this", it's evaluating the current
  | state against a new attempt and rejecting it for a business rule.
  */
  await executeGrillmanTool(state, "get_meat", {}, store, "get-1");
  const s1 = await executeGrillmanTool(
    state,
    "season_meat",
    {},
    store,
    "season-attempt-A",
  );
  const s2 = await executeGrillmanTool(
    state,
    "season_meat",
    {},
    store,
    "season-attempt-B",
  );
  console.log("\nseason_meat (different keys) 1st call:", s1);
  console.log("season_meat (different keys) 2nd call:", s2);
  console.log(
    "Blocked by the business-rule guard (different results from each other)?",
    s1.ok === true && s2.ok === false ? "YES" : "NO",
  );

  /*
  | A REAL IDEMPOTENCY KEY
  | Same key on both calls, simulating a retry: the client doesn't know
  | whether the 1st call was processed (e.g. a network timeout before the
  | response arrived) and tries again with the SAME idempotencyKey. The
  | executor recognizes the already-seen key and returns the CACHED
  | result without running the handler again — the side effect
  | (seasoning the meat) isn't reapplied. Note that now both results are
  | IDENTICAL, unlike the test above.
  */
  const state2 = createInitialState("Test2", "medium");
  await executeGrillmanTool(state2, "get_meat", {}, store, "get-2");
  const retryKey = "season-attempt-C";
  const firstTry = await executeGrillmanTool(
    state2,
    "season_meat",
    {},
    store,
    retryKey,
  );
  const retryTry = await executeGrillmanTool(
    state2,
    "season_meat",
    {},
    store,
    retryKey,
  );
  console.log("\nseason_meat (same key, retry) 1st attempt:", firstTry);
  console.log("season_meat (same key, retry) 2nd attempt (cache):", retryTry);
  console.log(
    "Identical result via cache, without reapplying the side effect?",
    JSON.stringify(firstTry) === JSON.stringify(retryTry) ? "YES" : "NO",
  );

  /*
  | serve_meat (terminal, non-idempotent, retryable=false): without an
  | idempotency key, it's not safe to auto-retry this — serving twice on
  | a network failure, for example, would be a serious bug, not a minor
  | detail. With the idempotencyKey above, a real retry COULD be done
  | safely — but the executor still doesn't do this on its own, because
  | only the caller knows when something is actually a retry.
  */
  console.log(
    "\nserve_meat.retryable:",
    toolMetadata.serve_meat.retryable,
    "(that's why the executor never retries this Tool on its own)",
  );
}

async function main() {
  await testIdempotency();
  await runGrillmanAgent("Ana", "medium");
}

main();
