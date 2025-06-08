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

    for u in nodes:
        dist[u][u] = 0
        next_hop[u][u] = u
        for v in nodes:
            if weight_matrix.get(u, {}).get(v, math.inf) != math.inf:
                next_hop[u][v] = v

    # 三重迴圈鬆弛
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_hop[i][j] = next_hop[i][k]

    # 檢測負權重迴路
    for u in nodes:
        if dist[u][u] < 0:
            raise ValueError("Graph contains a negative-weight cycle")

    return dist, next_hop


# 範例用法
if __name__ == "__main__":
    # 以 adjacency list 表示的範例圖
    graph = {
        'A': [('B', 4), ('C', 2)],
        'B': [('C', 5), ('D', 10)],
        'C': [('E', 3)],
        'D': [('F', 11)],
        'E': [('D', 4)],
        'F': []
    }

    print("Dijkstra:", dijkstra(graph, 'A'))
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
    dist_matrix, next_hop = floyd_warshall(nodes, weight_matrix)
    print("Floyd-Warshall dist:", dist_matrix)
    print("Floyd-Warshall next:", next_hop)
