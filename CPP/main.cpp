#include <iostream>
#include <algorithm>
#include <vector>
#include <array>
#include <unordered_map>
#include <chrono>
#include <queue>
#include <cmath>
#include <limits>

using namespace std;

class IndexedPriorityQueue {
public:
    IndexedPriorityQueue() = default;

    void push(int destination, int source, int distance) {
        auto it = vertex_index.find(destination);
        if (it != vertex_index.end()) {
            int index = it->second;
            if (distance >= get<0>(heap[index]))
                return;
            get<0>(heap[index]) = distance;
            get<2>(heap[index]) = source;
            decrease_key(index);
        }
        else {
            distances[destination] = make_tuple(distance, destination, source);
            insert(destination);
        }
    }

    tuple<int, int, int> pop() {
        tuple<int, int, int> min_vertex = heap[0];
        pop_heap();
        return min_vertex;
    }

    bool empty() const {
        return heap.empty();
    }

    bool contains(int destination) const {
        return vertex_index.find(destination) != vertex_index.end();
    }

private:
    vector<tuple<int, int, int>> heap;
    unordered_map<int, tuple<int, int, int>> distances;
    unordered_map<int, int> vertex_index;

    void insert(int key) {
        heap.push_back(distances[key]);
        vertex_index[key] = (int)heap.size() - 1;
        push_heap();
    }

    void decrease_key(int index) {
        while (index > 0 && get<0>(heap[index]) < get<0>(heap[parent(index)])) {
            swap(index, parent(index));
            index = parent(index);
        }
    }

    void pop_heap() {
        swap(0, heap.size() - 1);
        distances.erase(get<1>(heap[heap.size() - 1]));
        vertex_index.erase(get<1>(heap[heap.size() - 1]));
        heap.pop_back();
        sift_down(0);
    }

    void push_heap() {
        int index = (int)heap.size() - 1;
        while (index > 0 && get<0>(heap[index]) < get<0>(heap[parent(index)])) {
            swap(index, parent(index));
            index = parent(index);
        }
    }

    void sift_down(int index) {
        int current = index;
        int heapSize = (int)heap.size();

        while (current < heapSize) {
            int left = left_child(current);
            int right = right_child(current);
            int smallest = current;

            if (left < heap.size() && get<0>(heap[left]) < get<0>(heap[smallest])) {
                smallest = left;
            }
            if (right < heap.size() && get<0>(heap[right]) < get<0>(heap[smallest])) {
                smallest = right;
            }

            if (smallest != current) {
                swap(current, smallest);
                current = smallest;
            }
            else {
                break;
            }
        }
    }

    void swap(size_t index_a, size_t index_b) {
        ::swap(heap[index_a], heap[index_b]);
        vertex_index[get<1>(heap[index_a])] = (int)index_a;
        vertex_index[get<1>(heap[index_b])] = (int)index_b;
    }

    static int parent(int index) {
        return (index - 1) / 2;
    }

    static int left_child(int index) {
        return (2 * index) + 1;
    }

    static int right_child(int index) {
        return (2 * index) + 2;
    }
};

struct VectorHash {
    size_t operator()(const vector<int>& V) const {
        size_t hash = V.size();
        for (auto& i : V) {
            hash ^= i + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

struct Graph {
    Graph() = default;
    Graph(int dim, int v_count, int e_count) :
            dimensions(dim), vertex_count(v_count), edge_count(e_count),
            uid_to_coord(v_count), coord_to_uid(v_count),
            uid_adj_list(v_count) {};
    int dimensions{}, vertex_count{}, edge_count{};
    unordered_map<vector<int>, int, VectorHash> coord_to_uid;
    vector<vector<int>> uid_to_coord;
    vector<vector<pair<int, int>>> uid_adj_list;
};

void print_int_vector(ostream& out, const vector<int>& vec) {
    out << "[";
    for (int i = 0; i < vec.size(); i++) {
        out << vec[i];
        if (i != vec.size() - 1) {
            out << ", ";
        }
    }
    out << "]";
}

void print_graph(const Graph& graph) {
    for (int i = 0; i < graph.vertex_count; i++) {
        cout << i << ": ";
        print_int_vector(cout, graph.uid_to_coord[i]);
        cout << "\n";
    }
    for (int i = 0; i < graph.vertex_count; i++) {
        cout << i << ": ";
        for (auto& j : graph.uid_adj_list[i]) {
            cout << "(" << j.first << ",";
            cout << j.second << ") ";
        }
        cout << "\n";
    }
}

int match_cost(const Graph& graph, const vector<int>& match) {
    int cost = 0;
    for (int i = 0; i < match.size(); i++) {
        if (match[i] < 0) {
            continue;
        }
        for (int j = 0; j < graph.uid_adj_list[i].size(); j++) {
            if (graph.uid_adj_list[i][j].first == match[i]) {
                cost += graph.uid_adj_list[i][j].second;
                break;
            }
        }
    }
    cost /= 2;
    return cost;
}

bool match_is_valid(const Graph& graph, const vector<int>& match) {
    vector<int> matched_count(graph.vertex_count, 0);
    for (int i = 0; i < graph.vertex_count; i++) {
        if (match[i] == -1) continue;
        matched_count[match[i]]++;
        if (matched_count[match[i]] > 1) {
            return false;
        }
        bool edge_found = false;
        for (auto edge : graph.uid_adj_list[i]) {
            if (edge.first == match[i]) {
                edge_found = true;
                break;
            }
        }
        if (!edge_found) {
            return false;
        }
    }
    return true;
}

bool read_input(Graph& graph) {
    int dim, v_count, e_count;
    cin >> dim >> v_count >> e_count;

    graph = Graph(dim, v_count, e_count);

    int uid_counter = 0;
    for (int i = 0; i < graph.vertex_count; i++) {
        vector<int> vertex_coord(graph.dimensions);
        for (int j = 0; j < graph.dimensions; j++) {
            cin >> vertex_coord[j];
        }
        graph.coord_to_uid.emplace(vertex_coord, uid_counter);
        graph.uid_to_coord[uid_counter] = vertex_coord;
        uid_counter++;
    }

    bool read_success = true;

    for (int i = 0; i < graph.edge_count; i++) {
        vector<int> first_coord(graph.dimensions);
        vector<int> second_coord(graph.dimensions);
        int weight;
        for (int j = 0; j < graph.dimensions; j++) {
            cin >> first_coord[j];
        }
        for (int j = 0; j < graph.dimensions; j++) {
            cin >> second_coord[j];
        }
        cin >> weight;

        auto first_uid_it = graph.coord_to_uid.find(first_coord);
        auto second_uid_it = graph.coord_to_uid.find(second_coord);
        if (first_uid_it == graph.coord_to_uid.end()) {
            cout.flush();
            cout << "\t[ERR] Unexpected vertex: ";
            print_int_vector(cout, first_coord);
            cout << "\n" << flush;
            read_success = false;
            continue;
        }
        if (second_uid_it == graph.coord_to_uid.end()) {
            cout.flush();
            cout << "\t[ERR] Unexpected vertex: ";
            print_int_vector(cout, second_coord);
            cout << "\n" << flush;
            read_success = false;
            continue;
        }

        int first_uid = (*first_uid_it).second;
        int second_uid = (*second_uid_it).second;

        graph.uid_adj_list[first_uid].emplace_back(second_uid, weight);
        graph.uid_adj_list[second_uid].emplace_back(first_uid, weight);
    }

    return read_success;
}

vector<bool> color_graph(const vector<vector<pair<int, int>>>& uid_adj_list) {
    vector<bool> visited(uid_adj_list.size(), false);
    vector<bool> colors(uid_adj_list.size());
    queue<int> q;
    q.push(0);
    colors[0] = true;
    while (!q.empty()) {
        int current = q.front();
        q.pop();
        visited[current] = true;
        for (pair<int, int> neighbor : uid_adj_list[current]) {
            if (visited[neighbor.first]) {
                continue;
            }
            colors[neighbor.first] = !colors[current];
            q.push(neighbor.first);
        }
    }
    return colors;
}

void augment_path(vector<int>& match, const vector<int>& previous, int head) {
    int cur_t = head;
    int cur_s = previous[head];
    while (true) {
        if (match[cur_s] >= 0) {
            match[match[cur_s]] = -1;
        }
        match[cur_s] = cur_t;
        match[cur_t] = cur_s;
        if (previous[cur_s] == -1) {
            break;
        }
        else {
            cur_t = previous[cur_s];
            cur_s = previous[cur_t];
        }
    }
}

void update_duals(const Graph& graph, const vector<bool>& color,
                  vector<int>& y, const vector<int>& distance, int t) {
    for (int i = 0; i < graph.vertex_count; i++) {
        if (distance[i] == -1) {
            continue;
        }
        int update_value = distance[t] - distance[i];
        if (update_value <= 0) {
            continue;
        }
        if (color[i]) {
            y[i] += update_value;
        }
        else {
            y[i] -= update_value;
        }
    }
}

bool hungarian_search(const Graph& graph, const vector<bool>& color, vector<int>& match, vector<int>& y) {
    // multi-source/target dijkstra
    vector<int> previous(graph.vertex_count, -1);
    vector<int> distance(graph.vertex_count, -1);
    IndexedPriorityQueue ipq;
    for (int i = 0; i < graph.vertex_count; i++) {
        if (color[i] && match[i] < 0) {
            ipq.push(i, -1, 0);
        }
    }
    int found_t = -1;
    while (!ipq.empty()) {
        tuple<int, int, int> q_top = ipq.pop();
        int dist = get<0>(q_top);
        int current_vertex = get<1>(q_top);
        int source_vertex = get<2>(q_top);
        distance[current_vertex] = dist;
        previous[current_vertex] = source_vertex;

        if (!color[current_vertex] && match[current_vertex] < 0) {
            if (found_t < 0) {
                found_t = current_vertex;
            }
            continue;
        }

        if (!color[current_vertex]) {
            if (distance[match[current_vertex]] < 0) {
                ipq.push(match[current_vertex], current_vertex, dist);
            }
            continue;
        }

        for (const pair<int, int>& neighbor : graph.uid_adj_list[current_vertex]) {
            if (distance[neighbor.first] >= 0 || neighbor.first == match[current_vertex]) {
                continue;
            }
            int slack = neighbor.second - y[current_vertex] - y[neighbor.first];
            int new_dist = dist + slack;
            ipq.push(neighbor.first, current_vertex, new_dist);
        }
    }

    if (found_t < 0) {
        // No path found. Matching is optimal
        return false;
    }

    update_duals(graph, color, y, distance, found_t);

    augment_path(match, previous, found_t);

    return true;
}

double score_separator(int vertex_count, int sep_size, int front_size, int back_size) {
    // Tunable heuristic values
    const double side_upper = (2 * vertex_count) / 3.0;
    const double side_lower = (vertex_count) / 12.0;
    const double separator_upper = (2 * sqrt(2)) * (sqrt(vertex_count));

    // other constants
    const double v_point = (side_upper - side_lower) / 2.0;
    const double isv_point = 1 / (v_point * v_point);

    double score = 0;
    if (sep_size <= separator_upper) {
        score += sep_size / separator_upper;
    }
    else {
        score += 1 + (sep_size - separator_upper);
    }
    if (front_size <= side_upper) {
        if (front_size < side_lower) {
            score += 1 + (side_lower - front_size);
        }
        else {
            score += isv_point * (front_size - v_point - side_lower) * (front_size - v_point - side_lower);
        }
    }
    else {
        score += 1 + (front_size - side_upper);
    }
    if (back_size <= side_upper) {
        if (back_size < side_lower) {
            score += 1 + (side_lower - back_size);
        }
        else {
            score += isv_point * (back_size - v_point - side_lower) * (back_size - v_point - side_lower);
        }
    }
    else {
        score += 1 + (back_size - side_upper);
    }
    return score;
}

vector<vector<int>> get_layer_size(const Graph& graph) {
    const int dimensions = graph.dimensions;
    const int vertex_count = graph.vertex_count;
    vector<vector<int>> cut(dimensions, vector<int>(1));
    for (int i = 0; i < vertex_count; i++) {
        for (int j = 0; j < dimensions; j++) {
            if (cut[j].size() <= graph.uid_to_coord[i][j]) {
                cut[j].resize(graph.uid_to_coord[i][j] + 1);
            }
            cut[j][graph.uid_to_coord[i][j]]++;
        }
    }
    return cut;
}

vector<int> get_separator(const Graph& graph) {
    vector<vector<int>> layer_size = get_layer_size(graph);

    int best_layer_dimension = -1;
    int best_layer_index = -1;
    double best_layer_score = numeric_limits<double>::max();

    for (int i = 0; i < layer_size.size(); i++) {
        int front_side_size = 0;
        int back_side_size = graph.vertex_count;
        for (int j = 0; j < layer_size[i].size(); j++) {
            back_side_size -= layer_size[i][j];

            double score = score_separator(graph.vertex_count, layer_size[i][j],
                                           front_side_size, back_side_size);
            if (score < best_layer_score) {
                best_layer_score = score;
                best_layer_dimension = i;
                best_layer_index = j;
            }

            front_side_size += layer_size[i][j];
        }
    }

    vector<int> separator;
    for (int i = 0; i < graph.vertex_count; i++) {
        if (graph.uid_to_coord[i][best_layer_dimension] == best_layer_index) {
            separator.push_back(i);
        }
    }

    return separator;
}

vector<int> new_graph_from_mask(const Graph& graph, Graph& new_graph, const vector<bool>& color,
                                vector<bool>& new_color, vector<bool>& mask) {
    int start = 0;
    while (start < graph.vertex_count && mask[start]) {
        start++;
    }
    if (start >= graph.vertex_count) {
        return {};
    }

    vector<int> include(graph.vertex_count, -1);
    int v_count_new = 0;
    vector<int> translation(0);

    // BFS to find vertices in the new graph
    queue<int> q;
    q.push(start);
    while (!q.empty()) {
        int current = q.front();
        q.pop();
        if (include[current] >= 0) {
            continue;
        }
        include[current] = v_count_new;
        translation.push_back(current);
        mask[current] = true;
        v_count_new++;
        for (pair<int, int> neighbor : graph.uid_adj_list[current]) {
            if (include[neighbor.first] > -1 || mask[neighbor.first]) {
                continue;
            }
            q.push(neighbor.first);
        }
    }

    new_graph = Graph(graph.dimensions, v_count_new, 0);
    //vector<int> translation(new_graph.vertex_count);
    new_color = vector<bool>(new_graph.vertex_count);

    for (int i = 0; i < graph.vertex_count; i++) {
        if (include[i] < 0) {
            continue;
        }
        new_graph.uid_to_coord[include[i]] = graph.uid_to_coord[i];
        new_graph.coord_to_uid.insert({ graph.uid_to_coord[i], include[i] });
        //translation[uid_counter] = i;
        new_color[include[i]] = color[i];

        for (pair<int, int> edge : graph.uid_adj_list[i]) {
            if (include[edge.first] < 0) {
                continue;
            }
            new_graph.edge_count++;
            new_graph.uid_adj_list[include[i]].emplace_back(include[edge.first], edge.second);
        }
    }

    new_graph.edge_count /= 2;

    return translation;
}

void match_separator(const Graph& graph, const vector<int>& separator, vector<bool>& color,
                     vector<int>& match, vector<int>& y) {
    // Update duals on the separator
    for (int i : separator) {
        // int min_dual
        // for each edge find the most negative slack
        for (int j = 0; j < graph.uid_adj_list[i].size(); j++) {
            const pair<int, int>& edge = graph.uid_adj_list[i][j];
            int slack = edge.second - y[i] - y[edge.first];
            if (slack < 0) {
                y[edge.first] = 0;
                if (match[edge.first] >= 0) {
                    match[match[edge.first]] = -1;
                    match[edge.first] = -1;
                }
            }
        }
    }

    // find augmenting paths and augment
    while (hungarian_search(graph, color, match, y)) {}

}

vector<int> lipton_tarjan(const Graph& graph, vector<bool>& color, vector<int>& y) {
    if (graph.vertex_count < 2) {
        return vector<int>(graph.vertex_count, -1);
    }
    else if (graph.vertex_count == 2) {
        if (color[0]) {
            y[0] = graph.uid_adj_list[0][0].second;
        }
        else {
            y[1] = graph.uid_adj_list[0][0].second;
        }
        return { 1, 0 };
    }
    else if (graph.vertex_count == 3) {
        vector<int> match(3, -1);
        hungarian_search(graph, color, match, y);
        return match;
    }

    vector<int> separator = get_separator(graph);
    vector<bool> mask(graph.vertex_count, false);
    for (int a : separator) {
        mask[a] = true;
    }
    vector<int> match(graph.vertex_count, -1);
    while (true) {
        Graph new_graph;
        vector<bool> new_color;
        vector<int> translation = new_graph_from_mask(graph, new_graph, color, new_color, mask);
        if (translation.empty()) {
            break;
        }
        vector<int> y_prime(new_graph.vertex_count, 0);
        vector<int> sub_match = lipton_tarjan(new_graph, new_color, y_prime);
        for (int j = 0; j < new_graph.vertex_count; j++) {
            y[translation[j]] = y_prime[j];
            if (sub_match[j] < 0) {
                continue;
            }
            match[translation[j]] = translation[sub_match[j]];
        }
    }

    match_separator(graph, separator, color, match, y);

    return match;
}

vector<int> lipton_tarjan(const Graph& graph) {
    vector<bool> color = color_graph(graph.uid_adj_list);
    vector<int> y(graph.vertex_count, 0);
    return lipton_tarjan(graph, color, y);
}

vector<int> hungarian(const Graph& graph) {
    vector<bool> color = color_graph(graph.uid_adj_list);
    vector<int> match(graph.vertex_count, -1);
    vector<int> y(graph.vertex_count, 0);

    while (hungarian_search(graph, color, match, y)) {}

    return match;
}

vector<int> hungarian_variant(const Graph& graph, int& cost) {
    vector<bool> color = color_graph(graph.uid_adj_list);
    color.resize(color.size() + 1, false);
    vector<int> match(graph.vertex_count + 1, -1);
    vector<int> y(graph.vertex_count + 1, 0);
    // for vertex in s
    int s = 0;
    while (s < graph.vertex_count && !color[s]) s++;
    while (s < graph.vertex_count) {
        // single-source multi-target dijkstra to a vertex in t
        int current_t = graph.vertex_count;
        match[current_t] = s;
        vector<int> dist(graph.vertex_count + 1, numeric_limits<int>::max());
        vector<int> prev(graph.vertex_count + 1, -1);
        vector<bool> used(graph.vertex_count + 1);
        while (match[current_t] != -1) {
            used[current_t] = true;
            const int cur_s = match[current_t];
            int delta = numeric_limits<int>::max();
            int next_t;
            for (pair<int, int> neighbor : graph.uid_adj_list[cur_s]) {
                int t = neighbor.first;
                if (used[t]) {
                    continue;
                }
                int slack = neighbor.second - y[cur_s] - y[t];
                if (slack < dist[t]) {
                    dist[t] = slack;
                    prev[t] = current_t;
                }
            }
            for (int j = 0; j < graph.vertex_count; j++) {
                if (color[j] || used[j]) {
                    continue;
                }
                if (dist[j] < delta) {
                    delta = dist[j];
                    next_t = j;
                }
            }
            for (int j = 0; delta != numeric_limits<int>::max() && j <= graph.vertex_count; j++) {
                if (color[j]) {
                    continue;
                }
                if (used[j]) {
                    y[match[j]] += delta;
                    y[j] -= delta;
                }
                else {
                    dist[j] -= delta;
                }
            }
            current_t = next_t;
        }
        for (int j; current_t != graph.vertex_count; current_t = j) {
            j = prev[current_t];
            match[current_t] = match[j];
            match[match[j]] = current_t;
        }
        s++;
        while (s < graph.vertex_count && !color[s]) s++;
    }
    match.resize(match.size() - 1);
    cost = -y[graph.vertex_count];
    return match;
}

bool solve() {
    Graph graph;
    bool read_success = read_input(graph);
    if (!read_success) {
        cout << "\t[ERR] Skipping test case." << endl;
        return false;
    }

    cout << "\tHungarian Method:\n\t===\n";
    int h_cost = 0;
    auto start_time = chrono::high_resolution_clock::now();
    // hungarian
    vector<int> h_result = hungarian_variant(graph, h_cost);
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    if (graph.vertex_count <= 50) {
        cout << "\t\t";
        print_int_vector(cout, h_result);
        cout << "\n";
    }
    cout << "\t\tMatching cost: " << h_cost << "\n";
    cout << "\t===\tTime: " << duration.count() << " ms." << endl;




    cout << "\n\tLipton-Tarjan Method:\n\t===\n";
    start_time = chrono::high_resolution_clock::now();
    // lipton-tarjan
    vector<int> lt_result = lipton_tarjan(graph);
    end_time = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    if (graph.vertex_count <= 50) {
        cout << "\t\t";
        print_int_vector(cout, lt_result);
        cout << "\n";
    }
    int lt_cost = match_cost(graph, lt_result);
    cout << "\t\tMatching cost: " << lt_cost << "\n";
    cout << "\t===\tTime: " << duration.count() << " ms.\n" << endl;




    if (h_cost != lt_cost) {
        cout << "[WARN] Test Failed." << endl;
        return false;
    }
    return true;
}

int main() {
    cin.tie(nullptr);
    ios_base::sync_with_stdio(false);

    cout << "--------------------\n";

    int tests;
    cin >> tests;

    int failed = 0;
    for (int i = 0; i < tests; i++) {
        cout << "Begin test case: " << i << "\n--------------------\n" << endl;
        if (!solve()) {
            failed++;
        }
        cout << "--------------------" << endl;
    }

    cout << "\n\nTotal tests:\t" << tests << "\n";
    cout << "Total success:\t" << tests - failed << "\n";
    cout << "Total failed:\t" << failed << "\n";

    return 0;
}
