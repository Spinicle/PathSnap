"""Train + compare >=5 models with GroupKFold by trip_id (Owner: B).
Log per model: config, CV macro F1 (mean/std), student_test top-1 / macro F1 / log loss,
train time, inference time -> results/model_comparison.csv. Fixed seeds.
"""
