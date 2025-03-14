source .venv/bin/activate.fish
git clone https://github.com/braceal/protein_search_evals.git
cd protein_search_evals
#uv pip install -U pip setuptools wheel
uv pip install -e .