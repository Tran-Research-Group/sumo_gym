# sumo_gym
Gym environments for SUMO AV simulators

# Contribution
## Project Structure
Follow this structure as an example.
```
poetry-demo
├── pyproject.toml
├── README.md
├── poetry_demo
│   └── __init__.py
└── tests
    └── __init__.py
```
## Branching
- `main`: release version of the package
- `dev`: active development branch
- `feat/*`: feature development branch from `dev`. Create this branch for each feature you add

```
main
├── dev
       └── feat/av
```

## Git commit messages
This is the template of a commit message
```<type>: <verb in present tense> details```
Allowed types are:

    feat (feature)
    fix (bug fix)
    docs (documentation)
    style (formatting, missing semi colons, …)
    refactor
    test (when adding missing tests)
    chore (maintain)

Therefore, for example, a commit message is something like ```feat: add multiple vehicles```.

# Coding Style

- Numpy docstrings
- PEP8 for code style
- Sphinx to generate Readme
- GitHub Flow for how to add features to code (branching, merging etc.) (link an example of its use in Readme)

# Package Management
Use `poetry`

First time you cloned the repo or whenever the package dependencies in the toml file is updated, run `poetry install`.

For adding package, use `poetry add`.