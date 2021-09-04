call black --config pyproject.toml .
call flake8
call pyright .
call python scripts/mypy_runner.py
call python scripts/check_copyright_notices.py
call interrogate -c pyproject.toml
bash scripts/run_all_tests.sh
