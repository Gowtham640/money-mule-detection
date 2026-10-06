"""Coordinated money-mule attack patterns.

Each attack is defined once, as a plan builder that returns the ordered list of
transactions the attack consists of (plus any setup such as a shared device or
a freshly opened account). The live runtime executes a plan step by step
through the normal ingestion pipeline; the legacy DataFrame functions at the
bottom apply the same plan to pandas frames for the original training code.

``labeled`` is the simulator's ground truth (the accounts the original
implementation marked ``is_fraud = 1``). It is used only to evaluate detection
and never as a detection input.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from backend.generator import CHANNELS

# START_DATE kept for any external references but NOT used for timestamps
START_DATE = datetime.now() - timedelta(days=30)


@dataclass
class AttackStep:
    sender: str
    receiver: str
    amount: float
    channel: str
    device_id: str = None  # None = sender's registered device


@dataclass
class AttackPlan:
    attack_type: str
    attack_name: str
    steps: list
    labeled: list
    new_accounts: list = field(default_factory=list)
    device_updates: dict = field(default_factory=dict)

    @property
    def participants(self):
        seen = []
        for step in self.steps:
            for acc in (step.sender, step.receiver):
                if acc not in seen:
                    seen.append(acc)
        return seen


@dataclass
class AttackContext:
    accounts: dict  # account_id -> {"created_epoch", "device_id", ...}
    active_ids: list
    now: float
    rng: random.Random
    idle_ids: set = field(default_factory=set)
    new_account_id: object = None  # callable returning the next free account id

    def age_days(self, account_id):
        created = self.accounts.get(account_id, {}).get("created_epoch", self.now)
        return max(self.now - created, 0.0) / 86400.0


# ── Participant selection ──────────────────────────────────────────
# Recruiters favour newly opened accounts. Selection depends only on account
# profile, never on detector output, so early warnings are not self-fulfilling.
NEW_ACCOUNT_DAYS = 60
NEW_ACCOUNT_WEIGHT = 3.0


def _pick(ctx, n, exclude=(), prefer=None):
    excluded = set(exclude)
    pool = [a for a in ctx.active_ids if a not in excluded]
    if len(pool) < n:
        raise ValueError(f"need {n} accounts, only {len(pool)} available")
    keyed = []
    for acc in pool:
        weight = NEW_ACCOUNT_WEIGHT if ctx.age_days(acc) < NEW_ACCOUNT_DAYS else 1.0
        if prefer is not None and acc in prefer:
            weight *= 4.0
        keyed.append((ctx.rng.random() ** (1.0 / weight), acc))
    keyed.sort(reverse=True)
    return [acc for _, acc in keyed[:n]]


def _channel(ctx):
    return ctx.rng.choice(CHANNELS)


# ── Plan builders ──────────────────────────────────────────────────
def plan_fan_in(ctx):
    mule = _pick(ctx, 1)[0]
    feeders = _pick(ctx, 5, exclude=[mule])
    exit_acc = _pick(ctx, 1, exclude=[mule, *feeders])[0]
    steps = [AttackStep(f, mule, ctx.rng.randint(20000, 50000), _channel(ctx)) for f in feeders]
    # Mule moves out 90% of what came in
    steps.append(AttackStep(mule, exit_acc, int(sum(s.amount for s in steps) * 0.9), _channel(ctx)))
    return AttackPlan("fan_in", "Fan-In Pattern", steps, [mule, *feeders])


def plan_fan_out(ctx):
    mule = _pick(ctx, 1)[0]
    recipients = _pick(ctx, 5, exclude=[mule])
    steps = [AttackStep(mule, r, ctx.rng.randint(20000, 50000), _channel(ctx)) for r in recipients]
    return AttackPlan("fan_out", "Fan-Out Pattern", steps, [mule, *recipients])


def plan_circular_ring(ctx):
    ring = _pick(ctx, 4)
    amount = ctx.rng.randint(20000, 40000)
    steps = [AttackStep(ring[i], ring[(i + 1) % len(ring)], amount, _channel(ctx)) for i in range(len(ring))]
    return AttackPlan("circular_ring", "Circular Transaction Ring", steps, list(ring))


def plan_velocity_chain(ctx):
    chain = _pick(ctx, 5)
    amount = ctx.rng.randint(50000, 80000)
    steps = []
    for i in range(len(chain) - 1):
        steps.append(AttackStep(chain[i], chain[i + 1], amount, _channel(ctx)))
        amount = int(amount * 0.9)
    return AttackPlan("velocity_chain", "High-Velocity Transfer Chain", steps, list(chain))


def plan_cross_channel_burst(ctx):
    mule = _pick(ctx, 1)[0]
    recipients = _pick(ctx, 4, exclude=[mule])
    amount = ctx.rng.randint(20000, 40000)
    steps = [AttackStep(mule, r, amount, CHANNELS[i % len(CHANNELS)]) for i, r in enumerate(recipients)]
    return AttackPlan("cross_channel_burst", "Cross-Channel Burst Behavior", steps, [mule, *recipients])


def plan_shared_device_cluster(ctx):
    cluster = _pick(ctx, 4)
    shared_dev = f"DX{ctx.rng.randint(100, 999)}"
    steps = [
        AttackStep(cluster[i], cluster[(i + 1) % len(cluster)], ctx.rng.randint(15000, 30000), _channel(ctx), shared_dev)
        for i in range(len(cluster))
    ]
    return AttackPlan(
        "shared_device_cluster", "Shared Device Cluster", steps, list(cluster),
        device_updates={acc: shared_dev for acc in cluster},
    )


def plan_behavioral_drift(ctx):
    acc = _pick(ctx, 1)[0]
    others = [a for a in ctx.active_ids if a != acc]
    steps = [AttackStep(acc, ctx.rng.choice(others), ctx.rng.randint(30000, 60000), _channel(ctx)) for _ in range(8)]
    return AttackPlan("behavioral_drift", "Sudden Behavioral Drift", steps, [acc])


def plan_early_volume_spike(ctx):
    # This attack is always new-account based: a freshly opened account
    new_id = ctx.new_account_id() if ctx.new_account_id else f"A_NEW_{ctx.rng.randint(1000, 9999)}"
    new_account = {
        "account_id": new_id,
        "created_epoch": ctx.now,
        "device_id": f"DV-{new_id}",
        "ip_address": f"192.168.{ctx.rng.randint(0, 255)}.{ctx.rng.randint(1, 254)}",
        "home_channel": _channel(ctx),
        "balance": 100000,
    }
    receivers = _pick(ctx, 6, exclude=[new_id])
    steps = [AttackStep(new_id, r, ctx.rng.randint(20000, 40000), _channel(ctx)) for r in receivers]
    return AttackPlan("early_volume_spike", "Early-Stage Volume Spike", steps, [new_id], new_accounts=[new_account])


def plan_smurfing(ctx):
    mule = _pick(ctx, 1)[0]
    receiver = _pick(ctx, 1, exclude=[mule])[0]
    steps = [AttackStep(mule, receiver, ctx.rng.randint(2000, 5000), _channel(ctx)) for _ in range(15)]
    return AttackPlan("smurfing", "Smurfing Pattern", steps, [mule, receiver])


def plan_dormant_activation(ctx):
    acc = _pick(ctx, 1, prefer=ctx.idle_ids or None)[0]
    others = [a for a in ctx.active_ids if a != acc]
    steps = [AttackStep(acc, ctx.rng.choice(others), ctx.rng.randint(40000, 70000), _channel(ctx)) for _ in range(6)]
    return AttackPlan("dormant_activation", "Dormant Account Activation", steps, [acc])


ATTACK_CATALOG = {
    "fan_in": {
        "name": "Fan-In Pattern",
        "description": "Five feeders push 20-50k each into one mule, which forwards 90% onward.",
        "builder": plan_fan_in,
    },
    "fan_out": {
        "name": "Fan-Out Pattern",
        "description": "One mule disperses 20-50k to five recipient accounts.",
        "builder": plan_fan_out,
    },
    "circular_ring": {
        "name": "Circular Transaction Ring",
        "description": "Four accounts pass the same 20-40k amount around a closed loop.",
        "builder": plan_circular_ring,
    },
    "velocity_chain": {
        "name": "High-Velocity Transfer Chain",
        "description": "50-80k hops down a five-account chain, shaving 10% per hop.",
        "builder": plan_velocity_chain,
    },
    "cross_channel_burst": {
        "name": "Cross-Channel Burst Behavior",
        "description": "One mule sends to four accounts, rotating UPI/NEFT/IMPS/ATM/Mobile.",
        "builder": plan_cross_channel_burst,
    },
    "shared_device_cluster": {
        "name": "Shared Device Cluster",
        "description": "Four accounts move onto one device and cycle 15-30k between them.",
        "builder": plan_shared_device_cluster,
    },
    "behavioral_drift": {
        "name": "Sudden Behavioral Drift",
        "description": "A quiet account suddenly sends eight 30-60k transfers.",
        "builder": plan_behavioral_drift,
    },
    "early_volume_spike": {
        "name": "Early-Stage Volume Spike",
        "description": "A brand-new account immediately pushes six 20-40k transfers.",
        "builder": plan_early_volume_spike,
    },
    "smurfing": {
        "name": "Smurfing Pattern",
        "description": "Fifteen small 2-5k structured transfers to one receiver.",
        "builder": plan_smurfing,
    },
    "dormant_activation": {
        "name": "Dormant Account Activation",
        "description": "An idle account wakes up and sends six 40-70k transfers.",
        "builder": plan_dormant_activation,
    },
}


def build_plan(attack_type, ctx):
    entry = ATTACK_CATALOG.get(attack_type)
    if entry is None:
        raise KeyError(f"unknown attack type: {attack_type}")
    return entry["builder"](ctx)


def attack_types():
    return [
        {"key": key, "name": value["name"], "description": value["description"]}
        for key, value in ATTACK_CATALOG.items()
    ]


# ── Legacy DataFrame interface (original training code / controller) ──
def _context_from_frame(accounts_df, preferred_ids=None):
    active = accounts_df[accounts_df.get("is_active", pd.Series(True, index=accounts_df.index)) != False]
    accounts = {}
    for row in accounts_df.itertuples(index=False):
        created = pd.Timestamp(getattr(row, "creation_time", datetime.now())).timestamp()
        accounts[str(row.account_id)] = {"created_epoch": created, "device_id": str(row.device_id)}
    counter = {"n": 0}

    def new_id():
        counter["n"] += 1
        return f"A_NEW_{random.randint(1000, 9999)}{counter['n']}"

    return AttackContext(
        accounts=accounts,
        active_ids=[str(a) for a in active["account_id"]],
        now=datetime.now().timestamp(),
        rng=random.Random(),
        idle_ids=set(str(p) for p in (preferred_ids or [])),
        new_account_id=new_id,
    )


def _apply_plan_to_frames(plan, accounts_df, transactions_df):
    base_time = datetime.now()
    for account in plan.new_accounts:
        row = {
            "account_id": account["account_id"],
            "creation_time": datetime.fromtimestamp(account["created_epoch"]),
            "device_id": account["device_id"],
            "ip_address": account["ip_address"],
            "balance": account["balance"],
            "channel": account["home_channel"],
            "is_fraud": 0,
            "is_active": True,
        }
        accounts_df = pd.concat([accounts_df, pd.DataFrame([row])], ignore_index=True)
    accounts_df = accounts_df.set_index("account_id")
    for acc, device in plan.device_updates.items():
        accounts_df.loc[acc, "device_id"] = device
    tx_id = int(base_time.timestamp() * 1000)
    rows = []
    for i, step in enumerate(plan.steps):
        if accounts_df.loc[step.sender, "balance"] < step.amount:
            accounts_df.loc[step.sender, "balance"] += step.amount
        rows.append({
            "transaction_id": f"T{tx_id + i}",
            "sender": step.sender,
            "receiver": step.receiver,
            "amount": step.amount,
            "timestamp": base_time + timedelta(minutes=i),
            "channel": step.channel,
            "device_id": step.device_id or accounts_df.loc[step.sender, "device_id"],
            "ip_address": accounts_df.loc[step.sender, "ip_address"],
            "is_attack": True,
        })
        accounts_df.loc[step.sender, "balance"] -= step.amount
        accounts_df.loc[step.receiver, "balance"] += step.amount
    for acc in plan.labeled:
        accounts_df.loc[acc, "is_fraud"] = 1
    accounts_df = accounts_df.reset_index()
    transactions_df = pd.concat([transactions_df, pd.DataFrame(rows)], ignore_index=True)
    return accounts_df, transactions_df, plan.attack_name, base_time


def _legacy(attack_type):
    def run(accounts_df, transactions_df, preferred_ids=None):
        print(f"\n[Attack] {ATTACK_CATALOG[attack_type]['name']}")
        ctx = _context_from_frame(accounts_df, preferred_ids)
        plan = build_plan(attack_type, ctx)
        return _apply_plan_to_frames(plan, accounts_df, transactions_df)

    run.__name__ = f"{attack_type}_attack"
    return run


fan_in_attack = _legacy("fan_in")
fan_out_attack = _legacy("fan_out")
circular_ring_attack = _legacy("circular_ring")
velocity_chain_attack = _legacy("velocity_chain")
cross_channel_burst_attack = _legacy("cross_channel_burst")
shared_device_cluster_attack = _legacy("shared_device_cluster")
behavioral_drift_attack = _legacy("behavioral_drift")
early_volume_spike_attack = _legacy("early_volume_spike")
smurfing_attack = _legacy("smurfing")
dormant_activation_attack = _legacy("dormant_activation")

# ── Registry ───────────────────────────────────────────────────────
attack_registry = [
    fan_in_attack,
    fan_out_attack,
    circular_ring_attack,
    velocity_chain_attack,
    cross_channel_burst_attack,
    shared_device_cluster_attack,
    behavioral_drift_attack,
    early_volume_spike_attack,
    smurfing_attack,
    dormant_activation_attack,
]
