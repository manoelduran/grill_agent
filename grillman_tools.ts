import { z } from "zod";

/*
╔══════════════════════════════════════════════════════════════════════════╗
║ GRILLMAN — WORLD STATE + TOOLS                                            ║
╚══════════════════════════════════════════════════════════════════════════╝
| This file is the boundary between the LLM and Grillman's real world: the
| shape of the state Tools read/mutate, the Tool Schema sent to the model,
| the Tool Metadata only the runtime sees, and the executor that validates
| before running anything.
|
| Both grillman.ts (Agent Loop) and grillman_with_context_and_memory.ts
| (Context & Memory) import from here — the "world" and the actions
| possible in it are the same in both; what changes between the files is
| how the loop decides and what it remembers, not what the Tools do.
*/

/*
╔══════════════════════════════════════════════════════════════════════════╗
║ STATE                                                                     ║
╚══════════════════════════════════════════════════════════════════════════╝
| The Agent doesn't keep memory "in its head". It observes an explicit,
| external state that represents:
|
| - the customer's order (what was ordered, whether it's done)
| - the grill (temperature, whether it's on)
| - the meat (whether it's been grabbed, seasoned, which side, current
|   doneness)
|
| Every Tool READS and/or MUTATES this state. The LLM never sees this
| object directly — it only sees what Tools return as an observation.
*/
export type Doneness = "raw" | "rare" | "medium" | "well done";

export interface GrillmanState {
  order: {
    customer: string;
    desiredDoneness: Doneness;
    completed: boolean;
  };
  grill: {
    temperature: number;
    on: boolean;
  };
  meat: {
    obtained: boolean;
    seasoned: boolean;
    onGrill: boolean;
    side: "A" | "B" | null;
    heatAccumulated: number;
    currentDoneness: Doneness;
  };
  coolingEventTriggered: boolean;
}

export function createInitialState(
  customer: string,
  desiredDoneness: Doneness,
): GrillmanState {
  return {
    order: { customer, desiredDoneness, completed: false },
    grill: { temperature: 220, on: true },
    meat: {
      obtained: false,
      seasoned: false,
      onGrill: false,
      side: null,
      heatAccumulated: 0,
      currentDoneness: "raw",
    },
    coolingEventTriggered: false,
  };
}

/*
| Accumulated-heat thresholds for advancing the meat's doneness.
| The hotter the grill at the moment of check_doneness, the more heat
| gets accumulated per check — meat gets done faster.
*/
const RARE_THRESHOLD = 3;
const MEDIUM_THRESHOLD = 6;
const WELL_DONE_THRESHOLD = 9;

function calculateDoneness(heat: number): Doneness {
  if (heat >= WELL_DONE_THRESHOLD) return "well done";
  if (heat >= MEDIUM_THRESHOLD) return "medium";
  if (heat >= RARE_THRESHOLD) return "rare";
  return "raw";
}

/*
╔══════════════════════════════════════════════════════════════════════════╗
║ TOOL METADATA — SIDE EFFECTS & IDEMPOTENCY                                ║
╚══════════════════════════════════════════════════════════════════════════╝
| This is NOT sent to the LLM. It's information the runtime uses to
| decide HOW a Tool can be executed (retry, idempotency cache).
|
| check_grill and check_doneness only READ the state: side-effect-free,
| idempotent, safe to retry. Every other Tool changes the real world —
| calling them twice with the same input does NOT produce the same state,
| so they're non-idempotent and non-retryable by default.
*/
export interface ToolMetadata {
  hasSideEffect: boolean;
  idempotent: boolean;
  retryable: boolean;
  maxAttempts: number;
}

export const toolMetadata: Record<string, ToolMetadata> = {
  check_grill: {
    hasSideEffect: false,
    idempotent: true,
    retryable: true,
    maxAttempts: 3,
  },
  get_meat: {
    hasSideEffect: true,
    idempotent: false,
    retryable: false,
    maxAttempts: 1,
  },
  season_meat: {
    hasSideEffect: true,
    idempotent: false,
    retryable: false,
    maxAttempts: 1,
  },
  put_on_grill: {
    hasSideEffect: true,
    idempotent: false,
    retryable: false,
    maxAttempts: 1,
  },
  flip_meat: {
    hasSideEffect: true,
    idempotent: false,
    retryable: false,
    maxAttempts: 1,
  },
  check_doneness: {
    hasSideEffect: false,
    idempotent: true,
    retryable: true,
    maxAttempts: 3,
  },
  remove_from_grill: {
    hasSideEffect: true,
    idempotent: false,
    retryable: false,
    maxAttempts: 1,
  },
  serve_meat: {
    hasSideEffect: true,
    idempotent: false,
    retryable: false,
    maxAttempts: 1,
  },
};

/*
| Standardized result of any Tool: the Agent ALWAYS observes this,
| never the raw `state`.
*/
export type ToolResult =
  | { ok: true; observation: string; [key: string]: unknown }
  | { ok: false; observation: string; error: string };

const EmptyInput = z.object({});
const SeasonMeatInput = z.object({ seasoning: z.string().optional() });

/*
╔══════════════════════════════════════════════════════════════════════════╗
║ TOOL IMPLEMENTATIONS                                                      ║
╚══════════════════════════════════════════════════════════════════════════╝
| Each handler receives the current state (closed over via closure) and
| returns an observation. Business rules (e.g. "can't season meat that's
| already seasoned") live here — they're not schema validation errors,
| they're domain errors, and that's why they become `{ ok: false }`
| instead of an exception: the Agent CAN and SHOULD decide what to do
| next based on this (replanning).
*/
function doCheckGrill(state: GrillmanState): ToolResult {
  return {
    ok: true,
    observation: `Grill at ${state.grill.temperature}°C, ${
      state.grill.on ? "on" : "off"
    }.`,
    temperature: state.grill.temperature,
    on: state.grill.on,
  };
}

function doGetMeat(state: GrillmanState): ToolResult {
  if (state.meat.obtained) {
    return {
      ok: false,
      observation: "Meat has already been grabbed.",
      error: "meat_already_taken",
    };
  }
  state.meat.obtained = true;
  return { ok: true, observation: "Meat grabbed and ready to season." };
}

function doSeasonMeat(
  state: GrillmanState,
  input: z.infer<typeof SeasonMeatInput>,
): ToolResult {
  if (!state.meat.obtained) {
    return {
      ok: false,
      observation: "No meat has been grabbed yet.",
      error: "no_meat",
    };
  }
  if (state.meat.seasoned) {
    return {
      ok: false,
      observation: "Meat is already seasoned.",
      error: "already_seasoned",
    };
  }
  state.meat.seasoned = true;
  return {
    ok: true,
    observation: `Meat seasoned${input.seasoning ? ` with ${input.seasoning}` : ""}.`,
  };
}

function doPutOnGrill(state: GrillmanState): ToolResult {
  if (!state.meat.seasoned) {
    return {
      ok: false,
      observation: "Meat hasn't been seasoned yet.",
      error: "not_seasoned",
    };
  }
  if (state.meat.onGrill) {
    return {
      ok: false,
      observation: "Meat is already on the grill.",
      error: "already_on_grill",
    };
  }
  if (!state.grill.on) {
    return {
      ok: false,
      observation: "Grill is off.",
      error: "grill_off",
    };
  }
  state.meat.onGrill = true;
  state.meat.side = "A";
  loseHeat(state);
  return { ok: true, observation: "Meat placed on the grill, side A." };
}

function doFlipMeat(state: GrillmanState): ToolResult {
  if (!state.meat.onGrill) {
    return {
      ok: false,
      observation: "Meat isn't on the grill.",
      error: "not_on_grill",
    };
  }
  state.meat.side = state.meat.side === "A" ? "B" : "A";
  loseHeat(state);

  /*
  | REPLANNING — UNEXPECTED EVENT
  | In real life, charcoal loses strength, wind cools the coals, etc.
  | Here we simulate that: the first time the meat is flipped, the grill
  | cools down sharply. The Agent wasn't warned about this ahead of time
  | — it only finds out by observing, and has to adjust the original plan.
  */
  if (!state.coolingEventTriggered) {
    state.coolingEventTriggered = true;
    state.grill.temperature = 110;
    console.log(
      "⚠️  EVENT: the grill cooled down unexpectedly while the meat was flipping.",
    );
  }

  return {
    ok: true,
    observation: `Meat flipped, now on side ${state.meat.side}.`,
  };
}

function loseHeat(state: GrillmanState): void {
  /*
  | Every action that takes time on the grill cools it down a bit.
  | Floor at 60°C — below the threshold used in doCheckDoneness (40°C) —
  | so cooking never fully stalls.
  */
  state.grill.temperature = Math.max(60, state.grill.temperature - 8);
}

function doCheckDoneness(state: GrillmanState): ToolResult {
  if (!state.meat.onGrill) {
    return {
      ok: false,
      observation: "Meat isn't on the grill.",
      error: "not_on_grill",
    };
  }
  const heatGain = Math.max(state.grill.temperature - 40, 0) / 40;
  state.meat.heatAccumulated += heatGain;
  state.meat.currentDoneness = calculateDoneness(state.meat.heatAccumulated);
  loseHeat(state);
  return {
    ok: true,
    observation: `Current doneness: ${state.meat.currentDoneness}.`,
    currentDoneness: state.meat.currentDoneness,
  };
}

function doRemoveFromGrill(state: GrillmanState): ToolResult {
  if (!state.meat.onGrill) {
    return {
      ok: false,
      observation: "Meat is no longer on the grill.",
      error: "not_on_grill",
    };
  }
  state.meat.onGrill = false;
  return {
    ok: true,
    observation: `Meat removed from the grill at ${state.meat.currentDoneness}.`,
  };
}

function doServeMeat(state: GrillmanState): ToolResult {
  if (state.meat.onGrill) {
    return {
      ok: false,
      observation: "Meat is still on the grill, remove it before serving.",
      error: "still_on_grill",
    };
  }
  if (state.meat.currentDoneness !== state.order.desiredDoneness) {
    return {
      ok: false,
      observation: `Current doneness (${state.meat.currentDoneness}) doesn't match the order (${state.order.desiredDoneness}).`,
      error: "wrong_doneness",
    };
  }
  state.order.completed = true;
  return {
    ok: true,
    observation: `Meat served to ${state.order.customer} at ${state.meat.currentDoneness}.`,
  };
}

/*
╔══════════════════════════════════════════════════════════════════════════╗
║ TOOL SCHEMA — sent to the LLM                                             ║
╚══════════════════════════════════════════════════════════════════════════╝
*/
export const grillmanTools: any[] = [
  {
    type: "function" as const,
    function: {
      name: "check_grill",
      description:
        "Checks the current temperature and whether the grill is on. No side effects.",
      parameters: { type: "object", properties: {}, required: [] },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "get_meat",
      description: "Grabs the meat for the order to start preparing it.",
      parameters: { type: "object", properties: {}, required: [] },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "season_meat",
      description: "Seasons the meat that's already been grabbed. Can only be done once.",
      parameters: {
        type: "object",
        properties: {
          seasoning: { type: "string", description: "Seasoning used, optional" },
        },
        required: [],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "put_on_grill",
      description: "Puts the already-seasoned meat on the grill.",
      parameters: { type: "object", properties: {}, required: [] },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "flip_meat",
      description: "Flips the meat that's on the grill to the other side.",
      parameters: { type: "object", properties: {}, required: [] },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "check_doneness",
      description:
        "Checks the current doneness of the meat that's on the grill. No side effects.",
      parameters: { type: "object", properties: {}, required: [] },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "remove_from_grill",
      description: "Removes the meat from the grill.",
      parameters: { type: "object", properties: {}, required: [] },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "serve_meat",
      description:
        "Serves the meat to the customer. Only works if the current doneness matches the order.",
      parameters: { type: "object", properties: {}, required: [] },
    },
  },
];

/*
╔══════════════════════════════════════════════════════════════════════════╗
║ IDEMPOTENCY KEY                                                           ║
╚══════════════════════════════════════════════════════════════════════════╝
| Different from the business-rule guards inside each `do*` handler (e.g.
| "meat is already seasoned"), which block a NEW action that would violate
| an invariant. An idempotency key solves a different problem: "this is
| the SAME attempt as before (a retry), not a new action" — and so it
| returns the ALREADY CACHED result without touching the handler.
|
| Only makes sense for NON-idempotent Tools (side effects). An idempotent
| Tool (check_grill, check_doneness) should actually be re-executed on
| every call — it doesn't change the world, and caching the result would
| return a stale read.
*/
export type IdempotencyStore = Map<string, ToolResult>;

/*
╔══════════════════════════════════════════════════════════════════════════╗
║ TOOL EXECUTOR                                                             ║
╚══════════════════════════════════════════════════════════════════════════╝
| Centralizes: name/input validation, metadata check, and calling the
| handler. The LLM never decides directly what happens — it only
| requests, and the runtime decides whether and how to execute it.
*/
export async function executeGrillmanTool(
  state: GrillmanState,
  name: string,
  rawArgs: unknown,
  idempotencyStore: IdempotencyStore,
  idempotencyKey: string,
): Promise<ToolResult> {
  const metadata = toolMetadata[name];

  if (!metadata) {
    /*
    | PERMISSION BOUNDARY IN ACTION
    | If the LLM "hallucinates" a Tool outside the list, the runtime
    | simply doesn't recognize it — there's no handler for it.
    */
    return {
      ok: false,
      observation: `Unknown tool: ${name}`,
      error: "unknown_tool",
    };
  }

  /*
  | CACHE HIT — this key was already processed for this Tool. We return
  | the frozen result from the first time, without re-running the
  | handler: the side effect isn't reapplied.
  */
  const cacheKey = `${name}:${idempotencyKey}`;
  if (!metadata.idempotent && idempotencyStore.has(cacheKey)) {
    return idempotencyStore.get(cacheKey)!;
  }

  /*
  | Tool-calling models often omit `arguments` when the Tool has no
  | required parameters. A Zod object schema requires an object — even
  | with every field optional — so we normalize here before validating.
  */
  const args = rawArgs ?? {};

  const result = await executeHandler(state, name, args);

  if (!metadata.idempotent) {
    idempotencyStore.set(cacheKey, result);
  }

  return result;
}

async function executeHandler(
  state: GrillmanState,
  name: string,
  args: unknown,
): Promise<ToolResult> {
  switch (name) {
    case "check_grill":
      EmptyInput.parse(args);
      return doCheckGrill(state);
    case "get_meat":
      EmptyInput.parse(args);
      return doGetMeat(state);
    case "season_meat":
      return doSeasonMeat(state, SeasonMeatInput.parse(args));
    case "put_on_grill":
      EmptyInput.parse(args);
      return doPutOnGrill(state);
    case "flip_meat":
      EmptyInput.parse(args);
      return doFlipMeat(state);
    case "check_doneness":
      EmptyInput.parse(args);
      return doCheckDoneness(state);
    case "remove_from_grill":
      EmptyInput.parse(args);
      return doRemoveFromGrill(state);
    case "serve_meat":
      EmptyInput.parse(args);
      return doServeMeat(state);
    default:
      return {
        ok: false,
        observation: `Unknown tool: ${name}`,
        error: "unknown_tool",
      };
  }
}
