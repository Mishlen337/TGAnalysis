from igraph import Graph
import numpy as np


def get_cluster_graph(graph, cluster_membership):
    # Step 1: Create a new graph to represent the clusters
    graph_cluster = Graph(directed=True)

    # Step 2: Add nodes (representing communities)
    for i in range(max(cluster_membership) + 1):
        graph_cluster.add_vertex(i)

    # Step 3: Add edges between communities if nodes from different communities are connected
    for edge in graph.es:
        u, v = edge.source, edge.target
        community_u = cluster_membership[u]
        community_v = cluster_membership[v]

        if community_u != community_v:  # Only add edge if nodes are in different communities
            # Check if the edge already exists between the communities, otherwise add it
            if not graph_cluster.are_adjacent(community_u, community_v):
                graph_cluster.add_edge(community_u, community_v, weight=edge['weight'])  # Default weight as 1 if not specified
            else:
                edge_id = graph_cluster.get_eid(community_u, community_v)
                graph_cluster.es[edge_id]['weight'] += edge['weight']  # If 'weight' in edge.attributes() else 1
    
    return graph_cluster


def filter_graph_by_year(graph: Graph, year: int) -> Graph:
    """
    Filter the graph by removing all edges that do not have posts in the specified year.

    Parameters:
    - graph: igraph.Graph
    - year: The year to filter the graph by

    Returns:
    - A new igraph.Graph object with edges filtered by the specified year
    """
    filtered_graph = graph.copy()
    for edge in filtered_graph.es:
        filtered_dates_views = [(date, view) for date, view in zip(edge['dates'], edge['views']) if date.year == year]
        if filtered_dates_views:
            edge['dates'], edge['views'] = zip(*filtered_dates_views)
        else:
            edge['dates'], edge['views'] = [], []

        edge['weight'] = len(edge['views'])
   
    filtered_graph.delete_edges([edge.index for edge in filtered_graph.es if edge['weight'] == 0])

    return filtered_graph


def local_pagerank_igraph_weighted(graph, start_node, alpha=0.85, max_iter=1000, tol=1e-6):
    """
    Compute the Local PageRank for a given start node in a weighted igraph graph.

    Parameters:
    - graph: igraph.Graph
    - start_node: The node where the random walk starts (node name)
    - alpha: Damping factor
    - max_iter: Maximum number of iterations
    - tol: Convergence tolerance

    Returns:
    - A dictionary with nodes as keys and their local PageRank as values
    """
    # Map node names to indices
    node_to_idx = {v['name']: idx for idx, v in enumerate(graph.vs)}
    idx_to_node = {idx: v['name'] for idx, v in enumerate(graph.vs)}
    
    N = graph.vcount()
    start_idx = node_to_idx[start_node]

    # Create the weighted adjacency matrix
    adjacency_matrix = np.zeros((N, N), dtype=float)
    for edge in graph.es:
        source = edge.source
        target = edge.target
        weight = edge["weight"] # if weight_attr in edge.attributes() else 1.0
        adjacency_matrix[source, target] = weight

    # Normalize rows to create the transition probability matrix
    row_sums = adjacency_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero for nodes with no outgoing edges
    adjacency_matrix = adjacency_matrix / row_sums[:, None]

    # Personalized teleportation vector
    p = np.zeros(N)
    p[start_idx] = 1.0

    # Initialize PageRank vector
    r = np.copy(p)

    for iteration in range(max_iter):
        r_new = alpha * (adjacency_matrix.T @ r) + (1 - alpha) * p
        # Check for convergence
        if np.linalg.norm(r_new - r, 1) < tol:
            break
        r = r_new

    # Convert to dictionary format
    local_pagerank = {idx_to_node[i]: float(rank) for i, rank in enumerate(r)}
    return local_pagerank
