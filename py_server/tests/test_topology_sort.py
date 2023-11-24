from mlcraft.utils import topology_sort


#############################################################
#   GUARANTEES TO WORK FOR DIRECTED GRAPHS WITH NO CYCLES   #
#############################################################


def CHECK(edges: dict[int, list[int]], all_nodes: list[int], ordered_nodes: list[int]):
    result_set = set(ordered_nodes)
    back_map = dict(zip(ordered_nodes, list(range(len(ordered_nodes)))))
    assert len(result_set) == len(ordered_nodes)
    assert result_set == set(all_nodes)
    for node_from in edges:
        for node_to in edges[node_from]:
            assert back_map[node_from] < back_map[node_to]


def test_simple():
    test_cases = (
        ([0], {0: [1], 1: [2], 2: [3]}, 4),
        ([0, 5], {0: [1], 1: [3], 5: [2], 2: [4]}, 6),
    )
    for start_nodes, edges, expected in test_cases:
        order = topology_sort(start_nodes, edges)
        CHECK(edges, list(range(expected)), order)


def test_empty():
    assert not topology_sort([], {})


def test_no_parallel():
    test_cases = (
        ([0, 1], {0: [2, 3, 4], 2: [5, 6], 4: [7], 1: [7]}, 8),
        ([0, 1], {0: [2, 3], 1: [2, 4], 3: [5], 4: [5]}, 6),
    )
    for start_nodes, edges, expected in test_cases:
        order = topology_sort(start_nodes, edges)
        CHECK(edges, list(range(expected)), order)


def test_parallel():
    test_cases = (
        (
            [0, 1],
            {
                0: [2, 3, 4],
                2: [5, 6],
                4: [7],
                1: [7, 8],
                5: [8],
                6: [8],
                3: [8],
                7: [8],
            },
            9,
        ),
        ([0, 1], {0: [2, 3], 1: [2, 4], 3: [5], 4: [5], 2: [5]}, 6),
        ([0, 6], {0: [1, 2], 1: [3], 2: [5], 3: [4], 4: [5], 5: [7], 6: [5, 7]}, 8),
    )
    for start_nodes, edges, expected in test_cases:
        order = topology_sort(start_nodes, edges)
        CHECK(edges, list(range(expected)), order)
