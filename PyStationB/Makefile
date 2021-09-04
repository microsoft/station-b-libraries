black::
	black --config pyproject.toml .

flake::
	flake8

mypy::
	python scripts/mypy_runner.py

pyright::
	pyright .

copyright::
	python scripts/check_copyright_notices.py

interrogate::
	interrogate -c pyproject.toml

pytest: pytest-linux pytest-windows pytest-cov

pytest-cov::
	@echo '***' Test coverage check NOT run. You might want to run this, but it will be slow, and it will happen in CI anyway:
	@echo '***  pytest --cov -n auto tests'

pytest-linux::
	bash scripts/run_all_tests.sh

pytest-windows::
	@echo '***' In an Anaconda prompt, run this:
	@echo '***  ' bash scripts/run_all_tests.sh

test: black flake pyright mypy copyright interrogate pytest

.PHONY: test
