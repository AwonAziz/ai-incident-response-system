# Architecture decision records

Short, dated notes on decisions that were not obvious, and what would change them.
Written so a reviewer can disagree with a specific decision instead of guessing
at it.

| ADR | Decision | Status |
| --- | --- | --- |
| [0001](docs/adr/0001-real-data-replay.md) | Evaluate on real labelled telemetry, not the simulator | Accepted |
| [0002](docs/adr/0002-evaluation-protocol.md) | Temporal split, training-quantile operating point, event-level metrics | Accepted |
| [0003](docs/adr/0003-dimensionless-features.md) | Detection features are dimensionless | Accepted |
| [0004](docs/adr/0004-drop-pandas-dependency.md) | The data layer stays stdlib-only | Accepted |
| [0005](docs/adr/0005-incident-persistence-and-auth.md) | Durable incident state (SQLite) + bearer-token API auth | Accepted |

## Template

```markdown
# NNNN. Title

* Status: proposed | accepted | superseded by [NNNN](...)
* Date: YYYY-MM-DD

## Context
What forced the decision.

## Decision
What we chose.

## Consequences
What gets better, what gets worse, what we give up.

## Revisit when
The condition that would invalidate this.
```