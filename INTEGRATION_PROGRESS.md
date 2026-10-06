# Money Mule Real-Time Integration — Progress

## Audit findings (pre-change state)
- Frontend `app.js` ran with `DEMO_SIM_MODE = true`: accounts, transactions, attacks, scores and stats were fabricated client-side (`buildDemoAccounts`, `buildDemoTransactionBatch`, 140 ms `runDemoRuntimeTick`). Backend was never called in this mode.
- Non-demo path polled 4 REST endpoints on timers; WS sent a full snapshot every 1 s.
- Frontend created backend accounts every 12 s via `setInterval(createNewAccount)`.
- No database; all state in pandas DataFrames, lost on restart.
- `model.py` hard-required CUDA, so it would not start on this Mac.
- Shipped RF flags 96.6% of normal active accounts on a live-like window; shipped GNN flags 0%. Both were trained on attack-only windows.
- `RealTimeEngine` leaked ground truth: `is_attack` added an "attack" signal (+0.58) and `is_fraud` forced risk to 0.97. Random jitter was added to risk.
- Attacks preferentially chose accounts the detector already flagged (circular "early warning").
- `adaptive_threshold_update` sign inverted (low recall raised the threshold).

## Checklist
- [ ] database (SQLite WAL)
- [ ] transaction persistence
- [ ] normal spawner (backend)
- [ ] attack spawner (progressive, backend)
- [ ] detection ingestion (single pipeline)
- [ ] models retrained on live feature distribution
- [ ] websocket (delta events, seq reconciliation)
- [ ] frontend store
- [ ] graph live updates
- [ ] transaction feed
- [ ] statistics
- [ ] charts
- [ ] investigation
- [ ] tests / browser verification
