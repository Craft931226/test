import math
from heapq import heappush, heappop


def dijkstra(graph, source):
    """
    graph: dict {u: [(v, w), ...]}
    """
    dist = {u: math.inf for u in graph}
    prev = {u: None for u in graph}
    dist[source] = 0
    visited = set()
    heap = [(0, source)]

    while heap:
        d_u, u = heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        for v, w in graph[u]:
            if w < 0:
                raise ValueError("Dijkstra 演算法不支援負權重")
            alt = d_u + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heappush(heap, (alt, v))

    return dist, prev


def bellman_ford(graph, source):
    """
    graph: dict {u: [(v, w), ...]}
    """
    dist = {u: math.inf for u in graph}
    prev = {u: None for u in graph}
    dist[source] = 0
    n = len(graph)

    # 放鬆所有邊 |V|-1 次
    for _ in range(n - 1):
        updated = False
        for u in graph:
            if dist[u] == math.inf:
                continue
            for v, w in graph[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u
                    updated = True
        if not updated:
            break

    # 檢測負環
    for u in graph:
        if dist[u] == math.inf:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                raise ValueError("Graph contains a negative-weight cycle")

    return dist, prev


def floyd_warshall(nodes, weight_matrix):
    """
    nodes: list 所有頂點標籤
    weight_matrix: dict-of-dicts, weight_matrix[u][v] = 權重或 math.inf
    """
    # 初始化
    dist = {u: {v: weight_matrix.get(u, {}).get(v, math.inf) for v in nodes} for u in nodes}
    next_hop = {u: {v: None for v in nodes} for u in nodes}

    # for u in nodes:
    #     dist[u][u] = 0
    #     next_hop[u][u] = u
    #     for v in nodes:
    #         if weight_matrix.get(u, {}).get(v, math.inf) != math.inf:
    #             next_hop[u][v] = v

    # 三重迴圈鬆弛
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    # next_hop[i][j] = next_hop[i][k]

    # 檢測負權重迴路
    for u in nodes:
        if dist[u][u] < 0:
            raise ValueError("Graph contains a negative-weight cycle")

    return dist

import heapq

def prim_mst(adj, start=0):
    """
    adj: list[list[tuple[int,int]]]  # adj[u] = [(v, weight), ...]
    start: starting vertex index
    Returns: (mst_edges, total_weight)
    """
    n            = len(adj)
    in_mst       = [False] * n
    key          = [float('inf')] * n   # key[v] = best edge weight crossing A | V−A
    parent       = [-1] * n
    key[start]   = 0
    pq           = [(0, start)]         # (weight, vertex)

    while pq:                            # while Q ≠ ∅   (slide 25)
        w, u = heapq.heappop(pq)         # u ← EXTRACT-MIN(Q)
        if in_mst[u]:
            continue                     # ignore obsolete entries (lazy-delete)
        in_mst[u] = True

        for v, wt in adj[u]:             # for each (u,v) ∈ Adj[u]
            if not in_mst[v] and wt < key[v]:
                key[v]    = wt           # DECREASE-KEY
                parent[v] = u
                heapq.heappush(pq, (wt, v))
                
    mst_edges = []
    total_cost =0
    for v in range(n):
        if parent[v] != -1:
            mst_edges.append((parent[v], v, key[v]))
            total_cost += key[v]
    return mst_edges, total_cost

graph = {
    'A': [('B', 4), ('C', 2)],
    'B': [('C', 5), ('D', 10)],
    'C': [('E', 3)], # 加, ('A', -3)有負權重
    'D': [('F', 11)],
    'E': [('D', 4)],
    'F': []
}

# print("Dijkstra:", dijkstra(graph, 'A'))
print("Bellman-Ford:", bellman_ford(graph, 'A'))

# 以 weight_matrix 表示的範例圖
nodes = ['A', 'B', 'C', 'D', 'E', 'F']
weight_matrix = {
    'A': {'B': 4, 'C': 2},
    'B': {'C': 5, 'D': 10},
    'C': {'E': 3},
    'E': {'D': 4},
    'D': {'F': 11},
}
dist_matrix= floyd_warshall(nodes, weight_matrix)
print("Floyd-Warshall dist:", dist_matrix)
# print("Floyd-Warshall next:", next_hop)

adj = [
[(1, 2), (3, 6)],        # 0 連到 1(權重2), 3(權重6)
[(0, 2), (2, 3), (3, 8), (4, 5)],  # 1 連到 0,2,3,4
[(1, 3), (4, 7)],        # 2 連到 1(3), 4(7)
[(0, 6), (1, 8), (4, 9)],# 3 連到 0,1,4
[(1, 5), (2, 7), (3, 9)] # 4 連到 1,2,3
]

# 從頂點 0 開始找 MST：
mst_edges, total_weight = prim_mst(adj, start=0)

print("MST edges:", mst_edges)
print("Total weight:", total_weight)