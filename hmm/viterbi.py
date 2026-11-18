"""HMM + Viterbi (Owner: C).
States: candidate arcs per point. Emission: classifier prob. Transition: network path
distance vs straight-line distance on directed arc graph (networkx).
Handle: no candidates, no path, gaps > max_gap_s (break/restart chain).
Tune decoder params on train/student_test only.
"""
