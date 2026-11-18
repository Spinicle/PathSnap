def route_continuity(arcs, trip) -> float:
    """Share of consecutive predicted arcs connected in the directed graph; skips None, collapses repeats, NaN if < 2 arcs."""
    sequence = []

    for arc_id in arcs:
        if arc_id is None:
            continue

        arc_id = int(arc_id)
        if not sequence or arc_id != sequence[-1]:
            sequence.append(arc_id)

    if len(sequence) < 2:
        return float("nan")

    arc_table = trip["arcs"]
    connected_pairs = 0

    for previous_arc, next_arc in zip(sequence, sequence[1:]):
        previous_end = int(arc_table.at[previous_arc, "v"])
        next_start = int(arc_table.at[next_arc, "u"])
        connected_pairs += previous_end == next_start

    return connected_pairs / (len(sequence) - 1)