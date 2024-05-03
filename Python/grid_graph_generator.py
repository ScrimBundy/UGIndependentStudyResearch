import random
from collections import deque


class Vertex:
    def __init__(self, coordinate):
        self.coordinate = tuple(coordinate)

    def __eq__(self, other):
        if isinstance(other, Vertex):
            return self.coordinate == other.coordinate
        return False


def get_neighbors(vertex):
    dimensions = len(vertex.coordinate)
    neighbors = []

    for i in range(dimensions):
        for diff in range(-1, 2, 2):
            neighbor_coord = list(vertex.coordinate)
            neighbor_coord[i] += diff
            neighbors.append(Vertex(neighbor_coord))

    return neighbors


class Graph:
    def __init__(self, dimension):
        self.dimension = dimension
        self.edge_count = 0
        self.vertices = {}
        self.adj_list = {}

    def add_vertex(self, vertex):
        if self.has_vertex(vertex):
            return
        self.vertices[vertex.coordinate] = vertex
        self.adj_list[vertex.coordinate] = []

    def add_edge(self, vertex1, vertex2, weight):
        self.adj_list[vertex1.coordinate].append((vertex2, weight))
        self.adj_list[vertex2.coordinate].append((vertex1, weight))
        self.edge_count += 1

    def has_vertex(self, vertex):
        return vertex.coordinate in self.vertices

    def has_edge(self, vertex1, vertex2):
        if not (self.has_vertex(vertex1) and self.has_vertex(vertex2)):
            return False
        for neighbor, _ in self.adj_list[vertex1.coordinate]:
            if neighbor.coordinate == vertex2.coordinate:
                return True
        return False

    def get_random_empty_neighbor(self, vertex):
        neighbors = get_neighbors(vertex)
        empty_neighbors = [neighbor for neighbor in neighbors if not self.has_vertex(neighbor)]
        if empty_neighbors:
            return random.choice(empty_neighbors)
        else:
            return None

    def is_single_connected_component(self):
        if not self.vertices:
            return False  # No vertices in the graph
        visited = set()
        start_vertex = next(iter(self.vertices.values()))  # Start BFS from any vertex
        queue = deque([start_vertex])
        while queue:
            current_vertex = queue.popleft()
            if current_vertex.coordinate in visited:
                continue
            visited.add(current_vertex.coordinate)
            for neighbor, _ in self.adj_list[current_vertex.coordinate]:
                if neighbor.coordinate not in visited:
                    queue.append(neighbor)
        return len(visited) == len(self.vertices)

    def to_string(self):
        min_coords = [min(vertex.coordinate[dim] for vertex in self.vertices.values()) for dim in
                      range(self.dimension)]

        printed_edges = set()

        for vertex, edges in self.adj_list.items():
            adjusted_vertex = tuple(vertex[dim] - min_coords[dim] for dim in range(len(vertex)))
            for edge in edges:
                adjusted_edge_vertex = tuple(
                    edge[0].coordinate[dim] - min_coords[dim] for dim in range(len(edge[0].coordinate)))
                smaller_vert = min(adjusted_vertex, adjusted_edge_vertex)
                larger_vert = max(adjusted_vertex, adjusted_edge_vertex)
                printed_edges.add((smaller_vert, larger_vert, edge[1]))

        result = ""
        vertex_count = len(self.vertices)
        result += f"{self.dimension} {vertex_count} {len(printed_edges)}\n"

        for vertex in sorted(self.vertices.keys()):
            adjusted_coords = tuple(vertex[dim] - min_coords[dim] for dim in range(len(vertex)))
            result += " ".join(map(str, adjusted_coords)) + "\n"

        printed_edges = sorted(printed_edges)
        for edge in printed_edges:
            result += f"{' '.join(map(str, edge[0]))} {' '.join(map(str, edge[1]))} {edge[2]}\n"

        return result


def try_add_pair(graph: Graph, vertex: Vertex, weight_1, weight_2):
    random_vertex_1 = graph.get_random_empty_neighbor(vertex)
    if random_vertex_1 is None:
        return None, None
    random_vertex_2 = graph.get_random_empty_neighbor(random_vertex_1)
    if random_vertex_2 is None:
        return None, None
    graph.add_vertex(random_vertex_1)
    graph.add_vertex(random_vertex_2)
    graph.add_edge(vertex, random_vertex_1, weight_1)
    graph.add_edge(random_vertex_1, random_vertex_2, weight_2)
    return random_vertex_1, random_vertex_2


def int_generator():
    return random.randint(0, 100)


def generate_graph(n, m, p):
    # n - dimension
    # m - pair count
    # p - probability of adding an extra edge
    graph = Graph(n)
    origin = Vertex([0] * n)
    second = random.choice(get_neighbors(origin))
    graph.add_vertex(origin)
    graph.add_vertex(second)
    graph.add_edge(origin, second, int_generator())

    openings_list = [origin, second]
    pair_count = 1
    while pair_count < m:
        v = random.choice(openings_list)
        v_1, v_2 = try_add_pair(graph, v, int_generator(), int_generator())
        if v_1 is None:
            openings_list = [vertex for vertex in openings_list if vertex != v]
            continue
        openings_list.append(v_1)
        openings_list.append(v_2)
        pair_count += 1

    already_checked = set()

    for vertex in graph.vertices.values():
        unconnected_neighbors = [neighbor for neighbor in get_neighbors(vertex)
                                 if graph.has_vertex(neighbor) and not graph.has_edge(vertex, neighbor)]
        for neighbor in unconnected_neighbors:
            smaller = min(vertex.coordinate, neighbor.coordinate)
            larger = max(vertex.coordinate, neighbor.coordinate)
            if (smaller, larger) in already_checked:
                continue
            if random.random() <= p:
                graph.add_edge(vertex, neighbor, int_generator())
            already_checked.add((smaller, larger))
    return graph


if __name__ == '__main__':
    graph = generate_graph(6, 5000, 1)
    #print(f"{graph.dimension} {len(graph.vertices)} {graph.edge_count}")
    #print(str(graph.is_single_connected_component()))
    #even_parity = 0
    #for vertex in graph.vertices.keys():
    #    count = 0
    #    for dim in vertex:
    #        count += dim
    #    if count % 2 == 0:
    #        even_parity +=1
    #print(f"Even parity: {even_parity}")
    print(graph.to_string())
