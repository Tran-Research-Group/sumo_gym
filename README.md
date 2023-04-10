# sumo_gym
Gym environments for SUMO Traffic simulator

## Project Structure
Follow the structure below.
```
sumo_gym
├── pyproject.toml
├── poetry.lock
├── README.md
├── sumo_gym
│       └── __init__.py
└── tests
│       └── __init__.py
│       └── test_<class/function name>.py
└── scripts
```
## Package Management
Use `poetry`

First time you cloned the repo or whenever the package dependencies in the toml file is updated, run `poetry install`.

For adding package, use `poetry add`. `poetry.lock` is automatically generated, so there is no need to edit it.

## Git
### Branching
- `main`: release branch of the package
- `dev`: active development branch
- `feat/*`: feature development branch from `dev`. Create this branch for each feature you add and merge back to `dev`.

For example, branching can be done in this way:

```
main
├── dev
       └── feat/av
```

### Commit messages
This is the template of a commit message
```<type>: <verb in present tense> details```
Allowed types are:

    feat (feature)
    fix (bug fix)
    docs (documentation)
    style (formatting, missing semi colons, …)
    refactor (refactoring, make sure everything works the same before and after)
    test (when adding missing tests)
    chore (maintain, package managements)

Therefore, for example, a commit message is something like ```feat: add multiple vehicles to the environment.```.

## Coding Style

- Numpy docstrings
- PEP8 for code style
- Sphinx to generate Readme
- GitHub Flow for how to add features to code (branching, merging etc.) (link an example of its use in Readme)

