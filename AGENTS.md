# Agent Code Configuration

## Core Principles

### Persistence & Autonomy
- Context window will auto-compact when approaching limits. **Never stop tasks early due to token concerns.**
- Before context refresh: save progress and state to memory.
- Work autonomously until complete. Do not artificially stop tasks regardless of remaining context.
- Monitor usage quota. Commit work regularly to avoid losing significant uncommitted changes.

### Task Execution
- Work systematically until full completion.
- Delegate to subagents **only** when a separate context window provides clear benefit.
- Avoid over-engineering. Make only requested or clearly necessary changes.
- Keep solutions simple and focused.
- **When uncertain, ask the user before executing/implementing.** Do not guess or assume.

## Code Quality Standards

### Before Any Code Change
1. **Read first, then act.** Always inspect relevant files before proposing edits.
2. Do not speculate about code you haven't seen.
3. If user references a file/path, **you must open and inspect it** before explaining or proposing fixes.
4. Search rigorously for key facts in the codebase.
5. Review style, conventions, and abstractions before implementing new features.

### Error Handling Policy
- **Do not add try/catch or error handling unless explicitly requested.**
- Errors must surface to reveal bugs—silent failures hide problems.
- Never substitute dummy data or fallback values without explicit approval.
- Let exceptions propagate naturally so issues become visible immediately.

### Implementation Guidelines
- Match existing code style and patterns.
- Prefer existing abstractions over creating new ones.
- Minimal diff: change only what's necessary.
- No speculative features or "nice-to-have" additions.

## Workflow

```
1. Understand → Read all relevant code and context
2. Plan → Identify minimal necessary changes
3. Implement → Follow existing patterns exactly
4. Verify → Ensure changes work as intended
5. Commit → Save work regularly to avoid loss
```

## Remember

- Be rigorous, persistent, and thorough.
- Simple > Complex. Explicit > Implicit.
- When in doubt, read more code before acting.
