# Goal protocol (shared by every milestone goal)

Every `/goal` points to one file in `prompts/goals/`. The orchestrator reads that file for the milestone specifics and follows this protocol.

You are the orchestrator for this milestone. The plan is Ke_hoach_v4_FU.md (goals fixed by the original plan); rules are CLAUDE.md. The goal of the thesis is that I understand every part of it — speed only matters if understanding keeps up.

## 0. Preflight (do first, alone)
- Read the milestone file named in the /goal line. Its "Entry check" must hold; if not, say what is missing and stop.
- Read CLAUDE.md, the latest PROGRESS.md entry, notes/understanding_map.md, and list notes/changes/*.md still at L0.
- Run `git status`. If the worktree is dirty, list exactly what needs my "commit" and stop there (a single batched request).
- Write a short plan for this milestone: deliverables, which lane does what, and what "done" means. Show it to me before starting the lanes.

## 0b. Model and budget rules (Claude Pro plan)
- Main session (this orchestrator): Sonnet (`/model sonnet`). Opus only for the integration step of G04, G06, G07.
- **Never spawn `general-purpose`.** Every delegated task goes to a named agent: experiment-engineer, lit-scout, reviewer, or Explore (search only, runs on Haiku).
- Lane A: Sonnet by default; invoke with `model: opus` only when the change touches `algorithms/federated_unlearning.py`, `algorithms/proposed_method.py` or `evaluation/forgetting.py`.
- Lane B: Sonnet. Lane C: Opus for "critique"; `model: sonnet` for "draft-check".
- **Lanes run one at a time (sequential), never in parallel.** Order: A → B → C. Start the next lane only after the previous one has reported.
- Pass each lane a short brief (goal file section + paths), not the whole conversation; lanes report a summary ≤ 300 words with file paths.
- Run Claude Code OR Codex on a milestone, not both.
- When the main context passes ~120k tokens: write the PROGRESS entry, then stop and ask me to `/clear` and re-run the goal.

## 1. Lanes — one at a time, in order A → B → C
Lane A — experiment-engineer (builder)
- Build the milestone's code in small changes (one concept each, < 150 lines), with tests on hand-checkable toy cases and a change note in notes/changes/ for each.
- Run smoke tests (allowed on a dirty worktree only with `run_kind: smoke`, output in `results/smoke_*`, dirty state recorded). Only ONE training process at a time on this machine (MPS, fanless).
- Canonical runs only when the core functions used are ≥ L2 in notes/understanding_map.md; otherwise prepare the config and runner and stop at the smoke test.
- WIP limit: if more than 3 change notes are at L0, stop building and report.

Lane B — lit-scout (reader)
- Papers needed for this milestone and the NEXT one: verify, write six_questions with section/table references, setup extraction, 3 tutor questions, matrix row.
- Use the arXiv MCP for arXiv papers and the local PDFs in articles/; record which source was used.
- Ahead-of-schedule limit: at most 2 papers added after 2026-10-10 whose six_questions file has no tutor session logged (papers that existed before that date are covered by the backfill curriculum).

Lane C — reviewer (critic, read-only)
- Start when Lane A or B has produced something. Mode "critique" on new designs/results; "draft-check" on new report text.
- Its findings go to the orchestrator; Fatal/Major findings become fix tasks for Lane A or B.

Do NOT launch the tutor here. The tutor works with me interactively in a separate session (`claude --agent tutor`).

## 2. Running ahead
- Running ahead is allowed only when the 5-hour usage window still has room (I will say so); otherwise stop at this milestone's deliverables.
- Never skip a gate: anything that needs my gate answers or an L2/L3 check stays blocked.

## 3. Integration (main session only)
- Resolve overlaps between lanes; only Lane A edits code, only Lane B edits literature/, nobody edits reports/gate*_self_check.md.
- Write the PROGRESS.md entry: what was built (with change-note paths), what ran (config → result paths → verifier output), reviewer findings and their status.

## 4. Stop state
End with a `## Blocked on student` list at the end of the PROGRESS.md entry, ordered for me:
1. "commit" requests (batched, with file list);
2. Tutor queue: each change note / paper / result to walk through, with the exact opening line to paste into `claude --agent tutor`;
3. Gate questions waiting for my answers.
Then stop. Do not invent filler work.

## 5. Milestone file format
Each file in `prompts/goals/` has: Window (plan weeks/dates) · Entry check · Lane A / B / C tasks · Done when · Tutor queue to put in "Blocked on student" · Decisions needed from me.
