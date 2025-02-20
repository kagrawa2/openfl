# Contributing

We welcome contributions from the community. There are several ways to contribute:
* Improvements in [documentation](https://openfl.readthedocs.io/en/latest/).
* Contributing to OpenFL's code-base: via bug-fixes or feature additions.
* Answering questions on our [discussions page](https://github.com/securefederatedai/openfl/discussions).
* Participating in our [roadmap](https://github.com/securefederatedai/openfl/blob/develop/ROADMAP.md) discussions.

We have a slack [channel](https://join.slack.com/t/openfl/shared_invite/zt-ovzbohvn-T5fApk05~YS_iZhjJ5yaTw) and we host regular [community meetings](https://github.com/securefederatedai/openfl#support).


## How to contribute code
### Step 1. Open an issue

Before you start making any changes, it is always good to open an [issue](https://github.com/securefederatedai/openfl/issues/new/choose) first (assuming one does not already exist), outlining your proposed changes. We can give you feedback, and potentially validate the proposed changes.

For minor changes (akin to a documentation or bug fix), proceed to opening a Pull Request (PR) directly.

### Step 2. Make code changes

To modify code, you need to fork the repository. Set up a development environment as covered in the section "Setup environment" below.

### Step 3. Create a Pull Request (PR)

Once the change is ready, open a PR from your branch in your fork, to the `develop` branch in [securefederatedai/openfl](https://github.com/securefederatedai/openfl). 

OpenFL follows standard recommendations for PR formatting. Make sure to use the [Pull Request Template](https://github.com/securefederatedai/openfl/tree/develop/.github/pull_request_template.md) to provide a clear description of your changes, motivation, and relevant details.

[How to write the perfect pull request](https://github.blog/2015-01-21-how-to-write-the-perfect-pull-request/).

### Step 4. Sign your work

Signoff your patch commits using your real name. We discourage anonymous contributions.

    Signed-off-by: Joe Smith <joe.smith@email.com>

If you set your `user.name` and `user.email` git configs, you can sign your commits using:
```bash
git commit --signoff -m <commit message>
```

Your signature [certifies](http://developercertificate.org/) that you wrote the patch, or, you otherwise have the right to pass it on as an open-source patch.

OpenFL is licensed under the [Apache 2.0 license](https://github.com/securefederatedai/openfl/blob/develop/LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

### Step 5. Code review and merge

Verify that your contribution passes all tests in our CI/CD pipeline. In case of any failures, look into the error messages and try to fix them.

![CI/CD](images/CI_details.png)

Meanwhile, a reviewer will review the pull request and provide comments. Post few iterations of
reviews and changes (depending on the complexity of the changes), the PR will be approved for merge.

## Setup environment

We recommend setting up a local dev environment. Clone your forked repo to your local machine and install the dependencies.

```shell
git clone https://github.com/YOUR_GITHUB_USERNAME/openfl.git
cd openfl
pip install -e .
pip install -r linters-requirements.txt
```

## Code style

OpenFL uses [ruff](https://github.com/astral-sh/ruff) to lint/format code and [precommit](https://pre-commit.com/) checks.

Run the following command at the **root** directory of the repo to show lint errors.

```
sh scripts/lint.sh
```

To autoformat the code, run the following command:

```
sh scripts/format.sh
```
You may need to resolve errors that could not be resolved by autoformatting.

### Docstrings
Since docstrings cannot be verified programmatically, if you do write/edit a docstring, make sure to check them manually. OpenFL docstrings should follow the conventions below:

A **class** or a **function** docstring may contain:
* A one-line description of the class/function.
* Paragraph(s) of detailed information.
* Usage examples wherever applicable.
* Detailed description of function arguments, return types and possible exceptions raised.

## Update documentation
To rebuild documentation, install packages:

```bash
pip install -r docs/requirements.txt
```

Next, run:
```bash
sphinx-build -b html docs docs/_build/html -j auto
```

You may disable notebook execution if that takes too long:
```bash
sphinx-build -b html -D nb_execution_mode=off docs docs/_build/html -j auto
```

The `-j auto` option controls build parallelism. You may replace `auto` with a number to specify the number of jobs to run in parallel.

Serve the documentation locally:
```bash
python -m http.server --directory docs/_build/html
```