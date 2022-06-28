# Checklist

## Procedure to create a new remote repository in GitHub
  
- [ ] Check if a repository associated to your project already exists.
- [ ] Click on `new`.
- [ ] Add Repository name.
- [ ] Add Description. Description must be the project's full name.
- [ ] Be sure the repository is set to `Private`.
- [ ] Flag `Add a README.md file`.
- [ ] Flag `Add .gitignore` and choose the desired programming language.
- [ ] Flag `Choose a license` and choose the desired license (choose MIT License
or have a look [here](https://choosealicense.com/) if unknown).
- [ ] Create repository.
- [ ] Add repository topic: [procedure](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics).
  Choose between:

  - ML (classical machine learning)
  - QML (quantum machine learning)
  - QTI (quantum technology initiative)
  - ST (classical statistics)

- [ ] Use correct branching methods for a clean GitHub tree
([docs](https://gist.github.com/stuartsaunders/448036/5ae4e961f02e441e98528927d071f51bf082662f)
and [example](https://nvie.com/posts/a-successful-git-branching-model/)). Tips:

  - use `main` branch for production-ready code only
  - create `develop` branch for the latest delivered development changes for the next release
  - create your development branch where each contributor works on a daily basis

See as an example of a ready to public repositry
[here](https://github.com/CERN-IT-GOV-INN/PyMandelbrot).

## Requirements for the README

The `README.md` file must contain the following sections.

- [ ] Description of the project
- [ ] How to install

  - definition of virtual environment used (`anaconda` or `venv`)
  - instruction to install the package (with `requirements.txt` or `setup.cfg` etc)
  - instruction how to run the code

- [ ] Quick start: minimal working example / tutorials / demos

## Requirements for the CODE

- [ ] `requirements.txt` or `environment.yaml`(for conda) or
`setup.cfg + pyproject.toml` or `setup.py`
([setuptools](https://setuptools.pypa.io/en/latest/)).
- [ ] `src/packagename` folder with source files.
- [ ] Formatting: production code must be formatted with
[Black](https://github.com/psf/black).
- [ ] Function annotations: augment all functions and modules with
[dosctrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html).

## Requirements before PUBLISHING

- [ ] `bibliography.md`: Zenodo link to external papers and datasets used.
- [ ] Semantic versioning: comply with
[semver.org](https://github.com/semver/semver/blob/master/semver.md) and
[apache.org](https://apr.apache.org/versioning.html) versioning rules.
- [ ] Documentation: using [readthedocs](https://docs.readthedocs.io/en/stable/tutorial/)
and [simple formatting rules](https://hplgit.github.io/teamods/sphinx_api/html/sphinx_api.html).
Follow instructions to build documentation with sphinx [here](./how_to_sphinx.md).
- [ ] Citation policy: how to use and cite the code (e.g. BibTex reference).
Learn how to generate the `CITATION.cff` file at the [how to release](./how_to_make_release.md#citing-the-software) instructions.
- [ ] Large files support: upload datasets and large files to our
[Zenodo](https://zenodo.org/communities/cern-it-gov-inn/) community.

## Final review

- [ ] Submit request to organization admins for repository review.
- [ ] Create new release and archive it to Zenodo. Follow instructions to make a
new release in the [how to release](./how_to_make_release.md) instructions.

