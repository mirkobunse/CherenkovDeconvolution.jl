# docker_tests/

Use Docker to test the Python wrapper in a fresh Ubuntu installation.

## Running tests in Docker

You can take out the Docker unit tests by calling `make`.

## Running tests locally

The wrapper must also be tested with an existing Julia installation which you might have locally.

```
python -m venv venv
venv/bin/pip install -e .
venv/bin/python docker_tests/tests.py
```
