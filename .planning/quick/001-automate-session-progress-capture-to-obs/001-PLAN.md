---
phase: quick-001
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - ~/.claude/vault-capture.sh
  - ~/.claude/CLAUDE.md
autonomous: true
requirements: [QT-001]

must_haves:
  truths:
    - "vault-capture.sh accepts --project, --summary-path, --description, --task-type, --branch, --commit and writes a structured note"
    - "Notes land in ~/.claude/ObsidianVault/Claude/00-Inbox/ with YAML frontmatter"
    - "brain-index.sh runs after note is written"
    - "CLAUDE.md instructs GSD workflows to call vault-capture.sh after executor completes"
  artifacts:
    - path: "~/.claude/vault-capture.sh"
      provides: "Shell script that reads SUMMARY.md and writes Obsidian inbox notes"
    - path: "~/.claude/CLAUDE.md"
      provides: "Updated global instructions with vault-capture integration guidance"
  key_links:
    - from: "~/.claude/vault-capture.sh"
      to: "~/.claude/ObsidianVault/Claude/00-Inbox/"
      via: "file write"
      pattern: "ObsidianVault/Claude/00-Inbox"
    - from: "~/.claude/vault-capture.sh"
      to: "~/.claude/brain-index.sh"
      via: "background subprocess"
      pattern: "brain-index.sh"
---

<objective>
Build vault-capture.sh and wire it into the GSD workflow so that each completed GSD task automatically writes a structured note to the Obsidian vault's inbox.

Purpose: Eliminate manual changelog writing after GSD tasks. Every completed task gets a searchable, indexed note in Obsidian automatically.
Output: Working shell script + updated CLAUDE.md instructions for GSD integration.
</objective>

<context>
@.planning/STATE.md
@.planning/quick/001-automate-session-progress-capture-to-obs/001-CONTEXT.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Build vault-capture.sh</name>
  <files>~/.claude/vault-capture.sh</files>
  <action>
Create ~/.claude/vault-capture.sh — a POSIX-compatible shell script that:

**Arguments (all optional with sensible defaults):**
- `--project NAME` — project name (default: basename of cwd or "unknown")
- `--task-type TYPE` — "quick" or "phase" (default: "quick")
- `--summary-path PATH` — path to SUMMARY.md to extract content from
- `--description TEXT` — short task description
- `--branch NAME` — git branch (default: current branch via `git rev-parse`)
- `--commit HASH` — commit hash (default: HEAD short hash via `git rev-parse`)

**Core logic:**

1. Parse args via while/case loop. For any missing defaults, auto-detect from git.

2. If `--summary-path` provided and file exists, extract content from SUMMARY.md:
   - Read the file content
   - Extract the one-liner (first bold line after the H1 title, or first non-empty non-heading line)
   - Extract "Accomplishments" or "What was done" section (look for `## Accomplishments` or `## What was done` heading, grab lines until next `##`)
   - Extract "Files Created/Modified" or "Key Files" section (same heading-scan approach)
   - Extract "Decisions Made" or "Key design decisions" section
   - If sections not found, fall back to first 30 lines of body content (skip frontmatter between `---` fences)

3. If no `--summary-path` or file missing, create a minimal note using just `--description` and available metadata.

4. Generate unique filename: `YYYY-MM-DD-{project}-{task_type}-$(date +%H%M%S).md` (timestamp suffix avoids collisions when multiple tasks complete the same day).

5. Write the note to `~/.claude/ObsidianVault/Claude/00-Inbox/{filename}`:
```yaml
---
date: YYYY-MM-DD
project: {project}
branch: {branch}
type: gsd-{task_type}
commit: {commit}
tags: [gsd, auto-capture, {project}]
---
```

Then markdown body:
```
# {description or one-liner from summary}

## What Was Built
{accomplishments/what-was-done content, or description fallback}

## Key Files
{files section content, or "See summary for details"}

## Decisions
{decisions content, or "No notable decisions"}

---
*Auto-captured by vault-capture.sh from GSD {task_type} task*
*Summary: {summary_path or "N/A"}*
```

6. After writing, run `~/.claude/brain-index.sh` in the background (`nohup ... &>/dev/null &`) so it doesn't block the caller.

7. Print confirmation: `vault-capture: wrote {filepath}` to stdout.

**Error handling:**
- If `~/.claude/ObsidianVault/Claude/00-Inbox/` doesn't exist, `mkdir -p` it
- If summary-path provided but file doesn't exist, log warning to stderr and continue with minimal note
- If brain-index.sh doesn't exist, log warning to stderr and skip indexing
- Exit 0 on success, exit 1 only on write failure

Make the script executable (`chmod +x`).
  </action>
  <verify>
    <automated>~/.claude/vault-capture.sh --project test-project --task-type quick --description "Test capture" --branch main --commit abc123f && ls -la ~/.claude/ObsidianVault/Claude/00-Inbox/ | grep test-project</automated>
  </verify>
  <done>vault-capture.sh exists, is executable, creates a properly formatted note in the Obsidian inbox with YAML frontmatter, and triggers brain-index.sh in the background. Handles missing summary gracefully.</done>
</task>

<task type="auto">
  <name>Task 2: Test with real SUMMARY.md and update CLAUDE.md</name>
  <files>~/.claude/CLAUDE.md</files>
  <action>
**Part A: Integration test with a real summary.**

Run vault-capture.sh against an actual project SUMMARY.md to verify extraction works end-to-end:

```bash
~/.claude/vault-capture.sh \
  --project flexa-pipeline \
  --task-type phase \
  --summary-path .planning/phases/03-quality-uplift/03-02-SUMMARY.md \
  --description "Grasp visual quality" \
  --branch fix/e2e-pipeline
```

Verify the output note has properly extracted accomplishments, files, and decisions from the real summary. Check the frontmatter is valid YAML. If extraction missed content, fix the parsing logic in vault-capture.sh.

After verifying, delete both test notes from 00-Inbox (the one from Task 1 verify and this one) so we don't leave test artifacts.

**Part B: Update ~/.claude/CLAUDE.md with vault-capture integration instructions.**

Read the current ~/.claude/CLAUDE.md. Add a new subsection under the existing "### On session end:" in the "## Session Lifecycle" section. Add these instructions:

Under "### On session end:" add a new item (keep existing items):
```
3. After GSD tasks complete (quick or phase), capture progress to Obsidian vault:
   ```bash
   ~/.claude/vault-capture.sh \
     --project "$(basename "$PWD")" \
     --task-type "{quick|phase}" \
     --summary-path "{path-to-SUMMARY.md}" \
     --description "{brief task description}" \
     --branch "$(git rev-parse --abbrev-ref HEAD)" \
     --commit "$(git rev-parse --short HEAD)"
   ```
```

Also add a new subsection after the existing "### GSD" auto-routing rule:

```
### Vault Capture — Automatic after GSD tasks
- **When:** After any GSD executor completes (quick task or phase plan execution)
- **How:** Call `~/.claude/vault-capture.sh` with project name, task type, and summary path
- **Why:** Auto-captures progress notes to Obsidian 00-Inbox for searchable project history
- **Coexists with:** Manual changelogs in 05-Meta/Changelogs/ (different purpose: curated handoffs vs. granular auto-captures)
```

Do NOT change anything else in CLAUDE.md. Use the Edit tool for surgical additions.
  </action>
  <verify>
    <automated>grep -q "vault-capture" ~/.claude/CLAUDE.md && echo "CLAUDE.md updated" && ! ls ~/.claude/ObsidianVault/Claude/00-Inbox/*test-project* 2>/dev/null && echo "Test artifacts cleaned"</automated>
  </verify>
  <done>vault-capture.sh correctly extracts content from real SUMMARY.md files. CLAUDE.md contains instructions for GSD workflows to call vault-capture.sh after executor completion. No test artifacts remain in the vault inbox.</done>
</task>

</tasks>

<verification>
1. `~/.claude/vault-capture.sh --help` or running with `--project x --description y` produces a note in 00-Inbox
2. Running with `--summary-path` pointing at a real SUMMARY.md extracts accomplishments/files/decisions
3. `grep "vault-capture" ~/.claude/CLAUDE.md` confirms integration instructions present
4. `file ~/.claude/vault-capture.sh` confirms executable
</verification>

<success_criteria>
- vault-capture.sh is executable and handles all argument combinations (with/without summary, with/without defaults)
- Notes written to ~/.claude/ObsidianVault/Claude/00-Inbox/ have valid YAML frontmatter and readable markdown body
- brain-index.sh is triggered in background after note write
- CLAUDE.md updated with clear instructions for GSD workflow integration
- Manual changelogs in 05-Meta/Changelogs/ are untouched (coexistence confirmed)
</success_criteria>

<output>
After completion, create `.planning/quick/001-automate-session-progress-capture-to-obs/001-SUMMARY.md`
</output>
