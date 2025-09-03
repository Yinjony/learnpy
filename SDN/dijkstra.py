#dist中存储的是到各个点的最短距离，result_list则为最短路径
def dijkstra(graph):
    # 获取图的节点数
    n = len(graph)
    result_list = [[] for _ in range(n)]
    for i in range(n):
        result_list[i].append(0)
    # 初始化最短路径距离，默认起点(0, 0)为0，其它为无穷大
    dist = [float('inf')] * n
    dist[0] = 0

    # 标记节点是否已经处理过
    visited = [False] * n

    for _ in range(n):
        # 选择当前未处理的最短路径节点
        u = -1
        for i in range(n):
            if not visited[i] and (u == -1 or dist[i] < dist[u]):
                u = i

        # 标记该节点已处理
        visited[u] = True

        # 更新与当前节点相邻的未处理节点的距离
        for v in range(n):
            if not visited[v] and graph[u][v] != 999:  # 有边且未访问
                #dist[v] = min(dist[v], dist[u] + graph[u][v])
                if dist[u]+graph[u][v]<dist[v]:
                    dist[v]=dist[u]+graph[u][v]

                    m=len(result_list[u])
                    for i in range(m):
                        if result_list[u][i]!=0:

                            result_list[v].append(result_list[u][i])
                    if u!=0:
                        result_list[v].append(u)
    for i in range(1,n):
        result_list[i].append(i)

    return result_list


# 仅作为示例，实际使用时采用上述封装的函数传入graph即可
graph = [
    [0, 1, 999, 999, 999],
    [1, 0, 1, 1, 999],
    [999, 1, 0, 999, 1],
    [999, 1, 999, 0, 999],
    [999, 999, 1, 999, 0]
]

# 调用Dijkstra算法计算从节点0到其他节点的最短路径（给出了节点0到其它所有节点的最短路径，规定具体两端后按需修改即可）
result = dijkstra(graph)
print(result)