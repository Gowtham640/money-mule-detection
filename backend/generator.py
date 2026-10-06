import pandas as pd
import random
import random as _random
from datetime import datetime, timedelta
from backend.config import NUM_ACCOUNTS, NUM_TRANSACTIONS
import numpy as np

CHANNELS   = ["UPI", "NEFT", "IMPS", "ATM", "Mobile"]
START_DATE = datetime.now() - timedelta(days=30)


def generate_accounts(num_accounts=NUM_ACCOUNTS, rng=None):
    random     = rng or _random
    accounts   = []
    today      = datetime.now()
    # Most customers use their own phone; ~6% share a device with another
    # account (household). Previously ~10 accounts per device, which made
    # "shared device" true for every account and useless as a signal.
    device_ids = []

    for i in range(num_accounts):
        if device_ids and random.random() < 0.06:
            device_id = random.choice(device_ids)
        else:
            device_id = f"D{str(len(device_ids) + 1).zfill(3)}"
        device_ids.append(device_id)
        account_id = f"A{str(i+1).zfill(4)}"
        rand = random.random()
        if rand < 0.20:
            age_days = random.randint(0, 30)
        elif rand < 0.70:
            age_days = random.randint(31, 365)
        else:
            age_days = random.randint(366, 365*5)

        accounts.append({
            "account_id":    account_id,
            "creation_time": today - timedelta(days=age_days),
            "device_id":     device_id,
            "ip_address":    f"192.168.{random.randint(0,255)}.{random.randint(1,254)}",
            "balance":       random.randint(5000, 500000),
            "channel":       random.choice(CHANNELS),
            "is_fraud":      0,
            "is_active":     True,
        })
    return pd.DataFrame(accounts)


def generate_normal_transactions(accounts_df, num_transactions=1):
    transactions = []
    active = accounts_df[accounts_df["is_active"] == True]
    if len(active) < 2:
        return pd.DataFrame(transactions)

    accounts     = active["account_id"].tolist()
    cluster_size = max(10, len(accounts) // 20)
    clusters     = [accounts[i:i+cluster_size] for i in range(0, len(accounts), cluster_size)]
    tx_counter   = random.randint(100000, 999999)

    for _ in range(num_transactions):
        cluster = random.choice(clusters)
        if random.random() < 0.80 and len(cluster) >= 2:
            sender_id   = random.choice(cluster)
            receiver_id = random.choice(cluster)
        else:
            sender_id   = random.choice(accounts)
            receiver_id = random.choice(accounts)

        if sender_id == receiver_id:
            continue

        sender_rows = accounts_df[accounts_df["account_id"] == sender_id]
        if sender_rows.empty:
            continue
        sender = sender_rows.iloc[0]
        amount = random.randint(500, 8000)
        if sender["balance"] < amount:
            continue

        transactions.append({
            "transaction_id": f"T{tx_counter}",
            "sender":         sender_id,
            "receiver":       receiver_id,
            "amount":         amount,
            "timestamp":      datetime.now(),
            "channel":        random.choice(CHANNELS),
            "device_id":      sender["device_id"],
            "ip_address":     sender["ip_address"],
            "is_attack":      False,
        })
        accounts_df.loc[accounts_df["account_id"] == sender_id,   "balance"] -= amount
        accounts_df.loc[accounts_df["account_id"] == receiver_id, "balance"] += amount
        tx_counter += 1

    return pd.DataFrame(transactions)