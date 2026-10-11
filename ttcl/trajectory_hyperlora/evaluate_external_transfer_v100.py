from ttcl.trajectory_hyperlora import evaluate_joint_domains_v93 as e
from pathlib import Path
e.OUT=Path("results/trajectory_hyperlora/external_transfer_v100_20261010")
e.main("cuda:0")
