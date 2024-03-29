# Developer Notes

## Introduction

This document serves as a guide to the development process of the NETT toolkit. It provides an overview of the toolkit's development workflow.

## Online Documentation Website

The documentation website is hosted on GitHub Pages and is available at [https://buildingamind.github.io/NewbornEmbodiedTuringTest/](https://buildingamind.github.io/NewbornEmbodiedTuringTest/). It grabs the latest documentation from the `docs` folder in the repository. It creates the documentation using Sphinx. The documentation is written in reStructuredText format and is located in the `docs/source` folder. 

All assets used by Sphinx are located in the `docs/source/_static` folder. The `index.rst` file in the `docs/source` folder is the main entry point for the documentation. Each section of the documentation is pulled in from the `index.rst` file in each subdirectory in `docs/source`.

The website is updated automatically when changes are pushed to the `main` branch. To see how this process works, please see the `.github/workflows/docs.yml` file.

The configuration for the website is defined in `conf.py`. The website uses the `ReadTheDocs` theme. It is possible to change the theme by modifying the `html_theme` variable in `conf.py`. The website currently uses three extensions: `sphinx.ext.autodoc`, `sphinx.ext.napoleon`, and `myst-parser`. The `autodoc` extension is used to automatically generate documentation from the docstrings in the source code. The `napoleon` extension is used to parse the Google-style docstrings. The `myst-parser` extension is used to parse markdown files. 

To build the documentation locally, you can run the following command:

```bash
sphinx-build -M html docs/source/ docs/build/
```

Subsequent builds can be done by running the following commands:
```bash
cd docs
make html
```

The documentation can be viewed by opening the `index.html` file in the `docs/build` folder in a web browser.
