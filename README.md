# parkinson_disease

![PyPI version](https://img.shields.io/pypi/v/parkinson_disease.svg)

desc

* [GitHub](https://github.com/memzero/parkinson_disease/) | [PyPI](https://pypi.org/project/parkinson_disease/) | [Documentation](https://memzero.github.io/parkinson_disease/)
* Created by [Walter Mangione](memzero) | GitHub [@memzero](https://github.com/memzero) | PyPI [@memzero](https://pypi.org/user/memzero/)
* MIT License

## Features

* TODO

## Documentation

Documentation is built with [Zensical](https://zensical.org/) and deployed to GitHub Pages.

* **Live site:** https://memzero.github.io/parkinson_disease/
* **Preview locally:** `just docs-serve` (serves at http://localhost:8000)
* **Build:** `just docs-build`

API documentation is auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

Docs deploy automatically on push to `main` via GitHub Actions. To enable this, go to your repo's Settings > Pages and set the source to **GitHub Actions**.

## Development

To set up for local development:

```bash
# Clone your fork
git clone git@github.com:your_username/parkinson_disease.git
cd parkinson_disease

# Install in editable mode with live updates
uv tool install --editable .
```

This installs the CLI globally but with live updates - any changes you make to the source code are immediately available when you run `parkinson_disease`.

Run tests:

```bash
uv run pytest
```

Run quality checks (format, lint, type check, test):

```bash
just qa
```

## Author

parkinson_disease was created in 2026 by Walter Mangione.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
