"""
Topological Engine for Transit Deserts Prototype
===============================================
Implements the synthetic hexagonal grid, transit routes, metric matrix generators,
symmetrization, and a pure-Python persistent homology (H1) solver to identify
topological holes (Transit Deserts) and trapped nodes.
"""

import numpy as np
import scipy.spatial.distance as dist
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional

# --- 1. Grid & Coordinates Generation ---
def generate_hex_grid(cols: int = 6, rows: int = 6, spacing: float = 400.0) -> List[Tuple[float, float]]:
    """
    Generates a flat-topped hexagonal grid of centroids.
    spacing is the distance between adjacent centroids in meters.
    """
    coords = []
    w = spacing
    h = spacing * (np.sqrt(3) / 2)
    for r in range(rows):
        for c in range(cols):
            x = c * w + (r % 2) * (w / 2)
            y = r * h
            coords.append((x, y))
    return coords

# --- 2. Synthetic Transit Network Definition ---
class SyntheticTransitNetwork:
    def __init__(self, spacing: float = 400.0):
        self.coords = generate_hex_grid(cols=6, rows=6, spacing=spacing)
        self.num_nodes = len(self.coords)
        
        # Perimeter nodes in loop order
        self.perimeter_nodes = [0, 1, 2, 3, 4, 5, 11, 17, 23, 29, 35, 34, 33, 32, 31, 30, 24, 18, 12, 6]
        
        # Build set of perimeter transit adjacent links
        self.perimeter_links = set()
        p_len = len(self.perimeter_nodes)
        for idx in range(p_len):
            u = self.perimeter_nodes[idx]
            v = self.perimeter_nodes[(idx + 1) % p_len]
            self.perimeter_links.add((min(u, v), max(u, v)))
            
        # Define physical transit stops (hubs) on the perimeter
        self.transit_stops = {
            0: "LRT North-West Stop",
            5: "LRT North-East Stop",
            30: "Bus South-West Terminal",
            35: "Bus South-East Terminal",
            12: "Jeepney West Hub",
            23: "Jeepney East Hub"
        }
        
    def get_proximity_to_nearest_stop(self) -> np.ndarray:
        """
        Computes Euclidean distance in meters from each grid cell centroid to the nearest transit stop.
        """
        stop_coords = np.array([self.coords[idx] for idx in self.transit_stops.keys()])
        node_coords = np.array(self.coords)
        dists = dist.cdist(node_coords, stop_coords, 'euclidean')
        return np.min(dists, axis=1)

    def generate_matrices(self, departure_time: str) -> Dict[str, np.ndarray]:
        """
        Generates pairwise symmetric distance matrices for the four metrics:
        'travel', 'transfers', 'modes', and 'wait' using a graph-shortest-path simulation.
        Depends on departure_time ('6AM', '7AM', '12PM', '6PM', '7PM') to simulate congestion.
        """
        # Time-of-day multipliers
        congestion_factors = {
            '6AM': 1.0,
            '7AM': 1.5,  # Morning Peak
            '12PM': 1.1, # Midday
            '6PM': 1.6,  # Evening Peak
            '7PM': 1.3
        }
        factor = congestion_factors.get(departure_time, 1.0)
        
        # We will build base graphs for travel, transfers, modes, wait
        g_travel = nx.Graph()
        g_transfers = nx.Graph()
        g_modes = nx.Graph()
        g_wait = nx.Graph()
        
        # Set of center nodes representing the isolated interior
        center_nodes = {14, 15, 20, 21}
        
        # 1. Identify adjacent grid cells (Euclidean distance <= 420m)
        coords_arr = np.array(self.coords)
        euclidean_dists = dist.squareform(dist.pdist(coords_arr, 'euclidean'))
        
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                d = euclidean_dists[i, j]
                if d <= 420.0:  # Hexagonal neighbors
                    is_perimeter_link = (min(i, j), max(i, j)) in self.perimeter_links
                    is_center_involved = (i in center_nodes) or (j in center_nodes)
                    
                    # Compute link parameters
                    if is_perimeter_link:
                        # Transit line link (LRT/Bus) - very fast, low friction
                        t_travel = 0.5 * factor
                        t_transfers = 0.05
                        t_modes = 0.05
                        t_wait = 0.5 * factor
                    else:
                        # Walking link or local paratransit
                        if is_center_involved:
                            # Center isolation penalty
                            t_travel = 15.0 * factor
                            t_transfers = 1.0
                            t_modes = 1.0
                            t_wait = 12.0 * factor
                        else:
                            # Normal grid link
                            t_travel = 5.0
                            t_transfers = 0.3
                            t_modes = 0.3
                            t_wait = 4.0
                            
                    g_travel.add_edge(i, j, weight=t_travel)
                    g_transfers.add_edge(i, j, weight=t_transfers)
                    g_modes.add_edge(i, j, weight=t_modes)
                    g_wait.add_edge(i, j, weight=t_wait)
                    
        # 2. Compute all-pairs shortest paths to build dense matrices
        travel_mat = np.zeros((self.num_nodes, self.num_nodes))
        transfers_mat = np.zeros((self.num_nodes, self.num_nodes))
        modes_mat = np.zeros((self.num_nodes, self.num_nodes))
        wait_mat = np.zeros((self.num_nodes, self.num_nodes))
        
        # Calculate shortest path lengths
        path_travel = dict(nx.all_pairs_dijkstra_path_length(g_travel))
        path_transfers = dict(nx.all_pairs_dijkstra_path_length(g_transfers))
        path_modes = dict(nx.all_pairs_dijkstra_path_length(g_modes))
        path_wait = dict(nx.all_pairs_dijkstra_path_length(g_wait))
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                # Fallback to visual distance if graph is disconnected (should not be)
                travel_mat[i, j] = path_travel.get(i, {}).get(j, 99.0)
                # Ensure transfer count is rounded logically
                transfers_mat[i, j] = np.ceil(path_transfers.get(i, {}).get(j, 4.0))
                modes_mat[i, j] = np.ceil(path_modes.get(i, {}).get(j, 3.0))
                wait_mat[i, j] = path_wait.get(i, {}).get(j, 99.0)
                
        # Symmetrize just in case
        for m in [travel_mat, transfers_mat, modes_mat, wait_mat]:
            np.minimum(m, m.T, out=m)
            
        return {
            'travel': travel_mat,
            'transfers': transfers_mat,
            'modes': modes_mat,
            'wait': wait_mat
        }

# --- 3. Pure-Python Persistent Homology (H1) Solver ---
def compute_h1_persistence(coords: List[Tuple[float, float]], dist_matrix: np.ndarray) -> List[Dict]:
    """
    Computes H1 (1-dimensional cycles/holes) persistent homology.
    Returns list of dicts with keys: 'birth', 'death', 'persistence', 'cycle' (list of nodes in order).
    """
    n = len(coords)
    
    # 1. Gather all unique edges and sort them by weight
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j, dist_matrix[i, j]))
    # Sort by weight (birth), then lexicographically
    edges.sort(key=lambda x: (x[2], x[0], x[1]))
    
    edge_to_idx = { (e[0], e[1]): idx for idx, e in enumerate(edges) }
    
    # 2. Gather all triangles (2-simplices)
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Weight of triangle is max of its edge weights
                w = max(dist_matrix[i, j], dist_matrix[j, k], dist_matrix[i, k])
                triangles.append((i, j, k, w))
    # Sort triangles by weight, then lexicographically
    triangles.sort(key=lambda x: (x[3], x[0], x[1], x[2]))
    
    # 3. Build boundary matrix d2 (triangles to edges)
    d2: List[List[int]] = []
    for t in triangles:
        i, j, k = t[0], t[1], t[2]
        e1 = edge_to_idx[(i, j)]
        e2 = edge_to_idx[(j, k)]
        e3 = edge_to_idx[(i, k)]
        boundary = sorted([e1, e2, e3])
        d2.append(boundary)
        
    # We also keep track of the cycles created by edges
    parent = list(range(n))
    def find(u):
        path = []
        while parent[u] != u:
            path.append(u)
            u = parent[u]
        for node in path:
            parent[node] = u
        return u
        
    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            parent[root_u] = root_v
            return True
        return False
        
    # Find creator edges (edges that create cycles)
    creator_edges = []
    for idx, e in enumerate(edges):
        if not union(e[0], e[1]):
            creator_edges.append(idx)
            
    # 4. Perform column reduction of d2 (over Z2)
    low_to_col = {}
    reduced_d2 = [col.copy() for col in d2]
    
    for col_idx in range(len(triangles)):
        col = reduced_d2[col_idx]
        while len(col) > 0:
            pivot = col[-1]
            if pivot in low_to_col:
                other_col_idx = low_to_col[pivot]
                other_col = reduced_d2[other_col_idx]
                new_col = list(set(col) ^ set(other_col))
                new_col.sort()
                reduced_d2[col_idx] = new_col
                col = new_col
            else:
                low_to_col[pivot] = col_idx
                break
                
    # 5. Extract persistence pairs
    holes = []
    max_val = np.max(dist_matrix) + 1.0
    
    for c_idx in creator_edges:
        birth = edges[c_idx][2]
        if c_idx in low_to_col:
            t_idx = low_to_col[c_idx]
            death = triangles[t_idx][3]
            cycle_edges = reduced_d2[t_idx]
            cycle_nodes = reconstruct_cycle(cycle_edges, edges)
            
            if death > birth and len(cycle_nodes) >= 3:
                holes.append({
                    'birth': birth,
                    'death': death,
                    'persistence': death - birth,
                    'cycle': cycle_nodes,
                    'creator_edge': edges[c_idx]
                })
        else:
            # Persistent hole
            cycle_nodes = find_cycle_in_graph(c_idx, edges, n)
            if len(cycle_nodes) >= 3:
                holes.append({
                    'birth': birth,
                    'death': max_val,
                    'persistence': max_val - birth,
                    'cycle': cycle_nodes,
                    'creator_edge': edges[c_idx]
                })
                
    # Sort holes by persistence descending
    holes.sort(key=lambda x: x['persistence'], reverse=True)
    return holes

def reconstruct_cycle(edge_indices: List[int], edges_list: List[Tuple[int, int, float]]) -> List[int]:
    """
    Given a list of edge indices in a cycle, order them into a sequential list of nodes.
    """
    if len(edge_indices) == 0:
        return []
    
    adj = {}
    for idx in edge_indices:
        u, v, _ = edges_list[idx]
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
        
    start = list(adj.keys())[0]
    path = [start]
    visited = {start}
    
    curr = start
    while True:
        neighbors = adj[curr]
        next_node = None
        for nbr in neighbors:
            if nbr not in visited:
                next_node = nbr
                break
        if next_node is not None:
            path.append(next_node)
            visited.add(next_node)
            curr = next_node
        else:
            if len(path) >= 3 and start in neighbors:
                break
            break
    return path

def find_cycle_in_graph(creator_edge_idx: int, edges_list: List[Tuple[int, int, float]], num_nodes: int) -> List[int]:
    g = nx.Graph()
    for i in range(creator_edge_idx):
        g.add_edge(edges_list[i][0], edges_list[i][1])
        
    u, v, _ = edges_list[creator_edge_idx]
    try:
        path = nx.shortest_path(g, source=u, target=v)
        return path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [u, v]

# --- 4. Point-In-Polygon Hole Trapping Check ---
def is_point_in_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    num = len(poly)
    j = num - 1
    c = False
    for i in range(num):
        if ((poly[i][1] > y) != (poly[j][1] > y)) and \
                (x < (poly[j][0] - poly[i][0]) * (y - poly[i][1]) / (poly[j][1] - poly[i][1] + 1e-9) + poly[i][0]):
            c = not c
        j = i
    return c

def get_trapped_nodes(holes: List[Dict], coords: List[Tuple[float, float]], threshold: float) -> Set[int]:
    trapped = set()
    for h in holes:
        if h['birth'] <= threshold < h['death']:
            poly_coords = [coords[idx] for idx in h['cycle']]
            for idx, pt in enumerate(coords):
                if idx not in h['cycle']:
                    if is_point_in_polygon(pt[0], pt[1], poly_coords):
                        trapped.add(idx)
    return trapped

# --- 5. Divergence Analysis ---
def evaluate_divergence(trapped_nodes: Set[int], proximity: np.ndarray, prox_threshold: float) -> List[str]:
    classes = []
    for idx in range(len(proximity)):
        is_far = proximity[idx] > prox_threshold
        is_trapped = idx in trapped_nodes
        
        if is_far and is_trapped:
            classes.append('True Desert')
        elif (not is_far) and is_trapped:
            classes.append('False Oasis')
        elif is_far and (not is_trapped):
            classes.append('False Desert')
        else:
            classes.append('Well-Served')
    return classes
