
from collections import deque

def print_all_path_from_source(graph, source, dest, visited, current_path) -> str:
    if (source == dest):
        print("[reached destination] through {}".format(current_path))
        return
    
    visited[source]=True

    for edge in graph.get_edges(source):
        if not edge.dest in visited:
            current_path.append(edge.dest)
            print_all_path_from_source(graph, edge.dest, dest, visited, current_path)
            current_path.pop()


def detect_cycle_directed_graph(graph, source, parent, already_visited, current_visiting) -> bool:
    """
    track 2 visited array
    1. for current path visited
    2. for global path visited
    """

    already_visited[source]=True
    current_visiting[source]=True

    for edge in graph.get_edges(source):
        if already_visited[edge.dest] is True:
            print("ohh boii!. we found a cycle")
            return True
        
        if current_visiting[edge.dest] is False:
            is_cycle=detect_cycle_directed_graph(graph, edge.dest, source, already_visited, current_visiting)
            if is_cycle:
                return is_cycle
            

    current_visiting[source]=False
    return False