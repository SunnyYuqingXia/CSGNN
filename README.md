# CSGNN

A standalone PyTorch implementation for node classification. This release provides a Texas demonstration; reproduction configurations for the other datasets will be made available later.

## Implementation

The implementation constructs a multi-hop structural target, learns a node-to-node matrix Z, and applies a low-rank and sparse reconstruction to obtain B for propagation. The reconstruction uses symmetric matrices, a warm start across outer iterations, and a largest-algebraic eigensolver with a fallback for convergence failures. A linear logit head is included in the model.

The model, data loader, training loop, and evaluation are contained in `main.py`. Dataset-specific configurations are defined in `PRESETS` near the beginning of the file.

## Installation

The code runs on CPUs. A reference environment uses Python 3.9 with:

```bash
pip install torch==1.10.2 numpy==1.23.5 scipy==1.9.3 networkx==2.8.4
```

## Data

Use the existing Texas Geom-GCN-format dataset and ten predefined splits. The program does not download data.

```text
CSGNN/
├── main.py
├── new_data/
│   └── <dataset>/
│       ├── out1_graph_edges.txt
│       └── out1_node_feature_label.txt
└── splits/
    └── <dataset>_split_0.6_0.2_<split>.npz
```

Each split file contains `train_mask`, `val_mask`, and `test_mask`. Split indices range from 0 to 9.

## Usage

Run one split:

```bash
python main.py --dataset texas --split 0 --data-root . --output results/texas/split0.json
```

If the data is stored elsewhere, set `--data-root` to the directory containing `new_data/` and `splits/`.

Run all ten splits sequentially in Bash:

```bash
for split in {0..9}; do
  python main.py --dataset texas --split "$split" --data-root . --output "results/texas/split${split}.json" || break
done
```

Texas uses a fixed configuration defined in `main.py`. Training runs for the configured number of epochs. The reported test accuracy corresponds to the epoch with the highest validation accuracy; validation loss breaks ties.

The Texas configuration uses a fixed seed of 42. 

The JSON field `best_val_test_acc` contains the selected test accuracy as a fraction between 0 and 1. Results and per-epoch metrics are saved when `--output` is specified.

To summarize ten completed splits:

```bash
python - <<'PY'
import json
from pathlib import Path
from statistics import mean, pstdev

dataset = 'texas'
scores = [100 * json.loads((Path('results') / dataset / f'split{i}.json').read_text())['best_val_test_acc'] for i in range(10)]
print(f'{dataset}: {mean(scores):.2f} ± {pstdev(scores):.2f}%')
PY
```

The standard deviation is computed over the ten split accuracies. Results can vary across numerical environments.
