import numpy as np
import igraph as ig
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_loghist(x, bins, ax=None):
  hist, bins = np.histogram(x, bins=bins)
  logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
  
  if ax is None:
    ax = plt.gca()

  ax.hist(x, bins=logbins)
  ax.set_xscale('log')
  ax.set_yscale('log')
  return ax


def visualize_graph(graph, ax, top_vertices, top_vertices_labels=None, size_coefficient=300):
    pagerank_values = graph.pagerank(weights=graph.es['weight'])

    # Use a segmented colormap that emphasizes higher values (e.g., 'plasma')
    colormap = plt.cm.get_cmap("tab20")
    vertex_colors = [colormap(cluster) for cluster in range(graph.vcount())]  # Red color for highest values

    node_sizes_pagerank = [size_coefficient * pr for pr in pagerank_values]
    layout = graph.layout_fruchterman_reingold(weights=1 / np.log1p(graph.es['weight']))

    # Assign labels only to the top vertices
    if top_vertices_labels:
        graph.vs['label'] = [top_vertices_labels[top_vertices.index(i)] if i in top_vertices else '' for i in range(graph.vcount())]
    else:
        graph.vs['label'] = [i if i in top_vertices else '' for i in range(graph.vcount())]

    ax = ig.plot(
        graph,
        target=ax,
        vertex_size=node_sizes_pagerank,
        # edge_arrow_size=1,  # Size of arrows to show edge direction
        edge_arrow_width=[5 if edge.source in top_vertices and edge.target in top_vertices else 0.1 for edge in graph.es],
        vertex_color=vertex_colors,  # Node color based on cluster
        vertex_label=graph.vs['label'],  # Add cluster id as label
        vertex_frame_width=0,  # Remove vertex frame
        vertex_frame_color=None,  # Remove vertex frame color
        edge_color=["red" if edge.source in top_vertices and edge.target in top_vertices else "grey" for edge in graph.es],  # Edge color
        edge_width=[1 if edge.source in top_vertices and edge.target in top_vertices else 0.1 for edge in graph.es],
        vertex_label_size=8,  # Set the label size to be smaller
        layout=layout,
    )


def visualize_heatmap(lp_df: pd.DataFrame, ax):
    # Assuming `lp_df` is a pandas DataFrame containing the Local PageRank values
    # `lp_df.T` is the transposed DataFrame, used for visualization in this example

    # Create a mask for the diagonal
    mask = np.eye(lp_df.shape[0], dtype=bool)

    sns.heatmap(
        lp_df, 
        annot=True, 
        cmap='coolwarm', 
        cbar=True, 
        linewidths=0.5, 
        mask=mask,
        vmin=0,
        vmax=0.25,
        ax=ax,
    )  # Apply the mask

    ax.xaxis.tick_top()
