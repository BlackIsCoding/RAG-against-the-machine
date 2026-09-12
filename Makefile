# ba9a run w debug


install:
	python3 -m pip install uv
	uv sync

lint:
	uv run flake8 src
	uv run mypy \
	--warn-return-any \
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs src

clean:
	@find . -type d \( -name "__pycache__" -o -name ".mypy_cache" \) -exec rm -rf {} +