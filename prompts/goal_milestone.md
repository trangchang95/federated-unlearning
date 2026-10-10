Month 4, Week 13–14: federated retrain gold standard M_retrain^(-A) + noise floor + forgetting metrics

You are the orchestrator for this milestone. The plan is Ke_hoach_v3_FU_AI_tutor.md; rules are CLAUDE.md. The goal of the thesis is that I understand every part of it — speed only matters if understanding keeps up.

## 0. Preflight (do first, alone)
- Read CLAUDE.md, the latest PROGRESS.md entry, notes/understanding_map.md, and list notes/changes/*.md still at L0.
- Run `git status`. If the worktree is dirty, list exactly what needs my "commit" and stop there (a single batched request).
- Write a short plan for this milestone: deliverables, which lane does what, and what "done" means. Show it to me before starting the lanes.

## 1. Lanes — run in parallel as background subagents
Lane A — experiment-engineer (builder)
- Build the milestone's code in small changes (one concept each, < 150 lines), with tests on hand-checkable toy cases and a change note in notes/changes/ for each.
- Run smoke tests. Only ONE training process at a time on this machine (MPS, fanless).
- Canonical runs only when the core functions used are ≥ L2 in notes/understanding_map.md; otherwise prepare the config and runner and stop at the smoke test.
- WIP limit: if more than 3 change notes are at L0, stop building and report.

Lane B — lit-scout (reader)
- Papers needed for this milestone and the NEXT one: verify, write six_questions with section/table references, setup extraction, 3 tutor questions, matrix row.
- Use the arXiv MCP for arXiv papers and the local PDFs in articles/; record which source was used.
- Ahead-of-schedule limit: at most 2 papers whose six_questions file has no tutor session logged.

Lane C — reviewer (critic, read-only)
- Start when Lane A or B has produced something. Mode "critique" on new designs/results; "draft-check" on new report text.
- Its findings go to the orchestrator; Fatal/Major findings become fix tasks for Lane A or B.

Do NOT launch the tutor here. The tutor works with me interactively in a separate session (`claude --agent tutor`).

## 2. Running ahead
- When every deliverable of this milestone is built and smoke-tested, Lanes A and B may start the next milestone's work, within the WIP limits above.
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
```