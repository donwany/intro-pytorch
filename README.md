## install
```bash
mkdir torch-tutorials
cd torch-tutorials

uv init --python 3.12
uv venv
source .venv/bin/activate

uv pip install torch=="2.2.0" torchvision torchaudio
uv add "numpy<2"

uv run python -c 'import torch; print(f"PyTorch version: {torch.__version__}")'
```