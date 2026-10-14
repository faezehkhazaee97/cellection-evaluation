
### Using uv

First, install uv if you haven't already:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or using pip
pip install uv
```

Then clone the repository and sync dependencies:

```bash
git clone git@github.com:hongruhu/cellection_CG_dev.git
cd cellection_CG_dev
uv sync
```

This will create a virtual environment and install all dependencies automatically.

Activate the virtual environment:
```bash
source .venv/bin/activate
```