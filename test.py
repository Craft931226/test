import math
from heapq import heappush, heappop


def dijkstra(graph, source):
    """
    Dijkstra 演算法（只支援非負權重）
    graph: dict {u: [(v, w), ...]}
    source: 起點頂點
    回傳: dist (距離字典), prev (前驅字典)
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
    Bellman-Ford 演算法（支援負權重，檢測負環）
    graph: dict {u: [(v, w), ...]}
    source: 起點頂點
    回傳: dist (距離字典), prev (前驅字典)
    若偵測到負權重迴路，拋出 ValueError
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
    Floyd–Warshall 演算法（All-Pairs Shortest Paths，支援負權重）
    nodes: list 所有頂點標籤
    weight_matrix: dict-of-dicts, weight_matrix[u][v] = 權重或 math.inf
    回傳: dist (巢狀字典距離), next_hop (巢狀字典路徑重建)
    若偵測到負權重迴路，拋出 ValueError
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

###############################################################################
# 1. Prim’s Greedy MST — adjacency-list version, O(E log V) with heapq
###############################################################################
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

def dijkstra_743():
    from collections import defaultdict
    import math, heapq
    times = [[1,2,1]]
    n = 2
    k = 2
    graph = defaultdict(list)
    for u,v,w in times:
        graph[u].append((v,w))
    for u in range(1, n+1):
        graph.setdefault(u, [])
    dist = {u: math.inf for u in graph}
    dist[k] = 0
    visited = set()
    heap = [(0, k)]

    while heap:
        d_u, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        for v, w in graph[u]:
            alt = d_u + w
            if dist[v] > alt:
                dist[v] = alt
                heapq.heappush(heap, (alt, v))
    if any(d == math.inf for d in dist.values()):
        print(-1)
    else:
        print(max(dist.values()))

def spanningtree():
    def prim_mst(adj, start=0):
        import heapq
        n = len(adj)
        in_mst = [False] * n
        key = [float('inf')] * n
        parent = [-1] *n
        key[start] = 0
        pq = [(0,start)]

        while pq:
            w, u = heapq.heappop(pq)
            if in_mst[u]:
                continue
            in_mst[u] = True
            for v, wt in adj[u]:
                if not in_mst[v] and wt < key[v]:
                    key[v] = wt
                    parent[v] = u
                    heapq.heappush(pq, (wt, v))
        total_cost = 0
        for v in range(n):
            if parent[v] != -1:
                total_cost += key[v]
        return total_cost
    points = [[0,0],[1,1],[1,0],[-1,1]]
    p_length = len(points)
    adj = []
    for i in range(p_length):
        each_point_adj = []
        for j in range(p_length):
            dst_x = abs(points[i][0] - points[j][0])
            dst_y = abs(points[i][1] - points[j][1])
            cost_edge = dst_x + dst_y
            each_point_adj.append((j, cost_edge))
        adj.append(each_point_adj)
    cost = prim_mst(adj)
    print(adj)
    print(cost)

def LCS():
    text1 = "abcde"
    text2 = "ace"

    m = len(text1)
    n = len(text2)
    c = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if text1[i] == text2[j]:
                c[i+1][j+1] = c[i][j] +1
            else:
                c[i+1][j+1] = max(c[i][j+1],c[i+1][j])
    print(c[m][n])
    
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