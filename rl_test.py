import random
import copy

def randomMincut(vertex_set, edge_set):
    while len(vertex_set) > 2:
        print('len(edge_set):', len(edge_set))
        random_index = random.randint(0, len(edge_set)-1)
        u, v = edge_set.pop(random_index)
        vertex_set.remove(u)
        vertex_set.remove(v)
        new_vertex = copy.copy(u)
        new_vertex.extend(v)
        vertex_set.append(new_vertex)
        new_edge = []
        for i in range(len(edge_set)):
            if edge_set[i][0] in [u, v] :
                edge_set[i][0] = new_vertex
            if edge_set[i][1] in [u, v]:
                edge_set[i][1] = new_vertex
            if edge_set[i][0] != edge_set[i][1]:
                new_edge.append(edge_set[i])
        edge_set = new_edge
    return vertex_set, len(edge_set)


if __name__ == '__main__':
    epoch = 100
    vertex_set, edge_set = [], []
    with open(r'data.txt', 'r') as f:
        for f_line in f.readlines():
            temp_line = f_line.split()
            vertex_set.append([temp_line[0]])
            for i in range(1, len(temp_line)):
                if [[temp_line[i]], [temp_line[0]]] not in edge_set:
                    edge_set.append([[temp_line[0]], [temp_line[i]]])
    # print(vertex_set)
    # print(edge_set)
    result = []
    for i in range(epoch):
        v = copy.deepcopy(vertex_set)
        e = copy.deepcopy(edge_set)
        rv, rw = randomMincut(v, e)
        # print(rv, rw)
        result.append((rv, rw))

    min_index = -1
    min_rw = 9999999
    for i, ([rv1, rv2], rw) in enumerate(result):
        if rw < min_rw:
            min_rw = rw
            min_index = i
    print('output:')
    [rv1, rv2], rw = result[min_index]
    print(rv1)
    print(rv2)
    print(rw)

