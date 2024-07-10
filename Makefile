.PHONY: clean lint test

clean:
	rm -rf .tox/
	rm -rf test-reports/

install-requirements-dev: clean
	python -m pip install -r requirements/dev.txt

requirements: clean
	python -m pip install -U pip-tools
	pip-compile requirements/base.in --output-file requirements/base.txt
	pip-compile requirements/dev.in --output-file requirements/dev.txt
	pip-compile requirements/test.in --output-file requirements/test.txt

test:
	tox --skip-pkg-install