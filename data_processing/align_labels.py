"""Per-point labels from .route (Owner: A).
.route is an ordered arc list, NOT one label per GPS point. Idea: monotonic alignment,
each point gets the nearest route arc at or after the previous point's arc.
TODO: implement, log alignment distance stats, document limitations in the report.
Output: DataFrame[trip_id, obs_idx, true_arc_id, align_dist_m]
"""
