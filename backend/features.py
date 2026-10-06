import pandas as pd
import networkx as nx
import tempfile
import os

# -----------------------------
# Build Transaction Graph
# -----------------------------

def build_transaction_graph(accounts_df, transactions_df):

    G = nx.DiGraph()

    # Add nodes (accounts)
    for _, row in accounts_df.iterrows():
        G.add_node(
            row["account_id"],
            is_fraud=row["is_fraud"]
        )

    # Add edges (transactions)
    for _, tx in transactions_df.iterrows():
        G.add_edge(
            tx["sender"],
            tx["receiver"],
            amount=tx["amount"],
            timestamp=tx["timestamp"],
            is_attack=tx.get("is_attack", False)
        )

    return G


# -----------------------------
# Visualize Graph
# -----------------------------

def visualize_fraud_subgraph(G):

    from pyvis.network import Network  # optional dependency, only for this export

    if G.number_of_nodes() == 0:
        return None

    net = Network(
        height="600px",
        width="100%",
        bgcolor="#111111",
        font_color="white",
        directed=True
    )

    net.barnes_hut()

    # Professional spacing
    net.repulsion(
        node_distance=220,
        central_gravity=0.2,
        spring_length=200,
        spring_strength=0.04
    )

    # Add nodes
    for node, data in G.nodes(data=True):

        is_fraud = data.get("is_fraud", 0)

        if is_fraud == 1:
            color = "red"
            size = 30
            title = f"""
            <b>Account ID:</b> {node}<br>
            <b>Status:</b> Fraud Account<br>
            <b>Role:</b> Suspicious Node<br>
            <b>Alert:</b> Part of coordinated mule network
            """
        else:
            color = "skyblue"
            size = 20
            title = f"""
            <b>Account ID:</b> {node}<br>
            <b>Status:</b> Normal Account
            """

        net.add_node(
            node,
            label=str(node),
            color=color,
            size=size,
            title=title,
            borderWidth=2,
            shape="dot",
            shadow=True
        )

    # Add edges
    for u, v, data in G.edges(data=True):

        amount = data.get("amount", 0)
        is_attack = data.get("is_attack", False)

        if is_attack:

            color = "red"
            width = 4
            title = f"""
            <b>⚠ Suspicious Transaction</b><br>
            Amount: {amount}
            """

        else:

            color = "#888888"
            width = 1.5
            title = f"""
            <b>Transaction</b><br>
            Amount: {amount}
            """

        net.add_edge(
            u,
            v,
            title=title,
            color=color,
            width=width,
            arrows="to"
        )



    net.set_options("""
    var options = {
    "physics": {
        "enabled": true,
        "barnesHut": {
        "gravitationalConstant": -8000,
        "centralGravity": 0.3,
        "springLength": 200,
        "springConstant": 0.04
        },
        "minVelocity": 0.75
    },
    "edges": {
        "arrows": {
        "to": {
            "enabled": true,
            "scaleFactor": 0.6
        }
        },
        "smooth": {
        "type": "dynamic"
        }
    }
    }
    """)

    # Save to temporary HTML
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp_file.name)

    return tmp_file.name


# -----------------------------
# Feature Engineering
# -----------------------------

def extract_node_features(accounts_df, transactions_df, G):

    feature_data = []
    account_lookup = accounts_df.set_index("account_id").to_dict("index")
    device_counts = accounts_df["device_id"].value_counts().to_dict()

    for node in G.nodes():

        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)

        incoming_edges = G.in_edges(node, data=True)
        outgoing_edges = G.out_edges(node, data=True)

        total_in_amount = sum(edge[2]["amount"] for edge in incoming_edges)
        total_out_amount = sum(edge[2]["amount"] for edge in outgoing_edges)

        # Retention ratio
        
        if total_in_amount > 0:
            retention_ratio = max(0,(total_in_amount - total_out_amount) / total_in_amount)
        else:
            retention_ratio = 0

        # Unique neighbors
        neighbors = set([edge[1] for edge in outgoing_edges] +
                        [edge[0] for edge in incoming_edges])
        unique_neighbors = len(neighbors)

        # Unique channels
        node_transactions = transactions_df[
            (transactions_df["sender"] == node) |
            (transactions_df["receiver"] == node)
        ]
        unique_channels = node_transactions["channel"].nunique()

        # Device cluster size
        device_id = account_lookup[node]["device_id"]


        device_cluster_size = device_counts.get(device_id, 1)


        # Transaction count
        transaction_count = node_transactions.shape[0]
        account_age_days = (
            pd.Timestamp.now() - account_lookup[node]["creation_time"]
        ).days

        feature_data.append({
            "account_id": node,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "total_in_amount": total_in_amount,
            "total_out_amount": total_out_amount,
            "retention_ratio": retention_ratio,
            "unique_neighbors": unique_neighbors,
            "unique_channels": unique_channels,
            "device_cluster_size": device_cluster_size,
            "transaction_count": transaction_count,
            "is_fraud": account_lookup[node]["is_fraud"],
            "account_age_days": account_age_days
        })

    features_df = pd.DataFrame(feature_data)

    return features_df