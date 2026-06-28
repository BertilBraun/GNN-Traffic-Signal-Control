# GNN Traffic Signal Control

## Remote Training Startup

On a fresh Ubuntu-style GPU node, start the first city training pass with:

```bash
curl -L https://raw.githubusercontent.com/BertilBraun/GNN-Traffic-Signal-Control/master/train.sh | bash
```

Default startup behavior:

- `DEVICE=cuda`
- `WORKERS=16`
- `RUN_FULL_PPO=0`, so it stops after the 100-iteration PPO pilot
- TensorBoard starts on `0.0.0.0:6006`
- city SUMO networks are rebuilt from the committed build recipes
- tests run before data collection and training

Common overrides:

```bash
curl -L https://raw.githubusercontent.com/BertilBraun/GNN-Traffic-Signal-Control/master/train.sh | env WORKERS=8 bash
curl -L https://raw.githubusercontent.com/BertilBraun/GNN-Traffic-Signal-Control/master/train.sh | env RUN_TESTS=0 bash
curl -L https://raw.githubusercontent.com/BertilBraun/GNN-Traffic-Signal-Control/master/train.sh | env RUN_FULL_PPO=1 bash
```

For live TensorBoard from your local machine:

```bash
ssh -L 6006:localhost:6006 user@node
```

Then open `http://localhost:6006`.
