# 0005. Durable incident state, and an authenticated control API

* Status: accepted
* Date: 2026-10-04

## Context

Two questions an interviewer asks about any on-call tool, and both had bad
answers:

1. *"Your pod gets rescheduled at 3am - what do you lose?"* Everything. The
   incident registry was a Python dict, so a restart lost the pages an operator
   was working through **and** the deduplication state that stops the same
   problem re-alerting.
2. *"How do you secure the API?"* It had no authentication at all. That is
   acceptable for a toy and unacceptable for something that can inject incidents
   and resolve them.

## Decision

**Persistence: SQLite, stdlib only, write-through, restore open incidents.**

* `src/triage/store.py` - one table for incidents, one for lifecycle events, WAL
  mode, a connection per operation (the control API serves requests on other
  threads and a shared connection would need cross-thread plumbing).
* `IncidentManager` takes an optional store. `add` / `acknowledge` / `resolve` /
  `escalate` write through; `touch` only bumps a counter and flushes every
  `persist_every`-th repeat, so a 40-minute alert storm is 4 writes, not 480.
* `restore()` brings **open** incidents back. Resolved history stays in the
  database for the record but is not resurrected: a restart should not re-page
  someone for a problem they already dismissed.
* `TriageEngine.bind()` seeds its dedup registry from whatever the manager holds,
  so a restored incident keeps suppressing its own repeat instead of re-alerting.
* Storage failures are logged and swallowed. Losing the audit trail must not stop
  detection - the tool's job is to keep watching the clouds.

**Authentication: a bearer token, `/health` exempt.**

* `API_TOKEN` (or `--api-token`). Compared with `hmac.compare_digest`, so a token
  is not leaked by response timing.
* Unset token means the API runs open and says so, once, in the log - fine for
  `localhost` and required by the Docker healthcheck, obvious anywhere else.
  `/health` stays public so liveness checks need no credentials.
* `WWW-Authenticate: Bearer` on 401, and the rejected count is exposed on
  `/health` so a misconfigured client is visible rather than mysterious.

## Consequences

* "What happens when it restarts?" now has a demonstrated answer: incidents,
  their state, their SLA deadlines and the dedup registry all come back, and the
  session summary reports how many were restored.
* One write-ahead database file, mounted next to the model in Docker, replacing a
  real class of "the dashboard was empty and I did not know why" support tickets.
* Two new knobs (`PERSISTENCE_ENABLED`, `RESTORE_ON_START`) and three CLI flags
  (`--no-persist`, `--database`, `--no-restore`) so the behaviour can be turned
  off in tests and demos without editing code.
* Writes are not transactional with respect to the notification: a crash between
  "incident persisted" and "Slack notified" can produce one repeat page after a
  restart. Accepted - the alternative (outbox table plus a delivery worker) is
  worth it only when notifications become billing-critical.

## Revisit when

Multiple pipeline replicas need to share incident state, which a local SQLite
file cannot do. At that point the store interface (`save` / `open_incidents` /
`append_event`) is the seam: a Postgres implementation is a new class, not a
rewrite.