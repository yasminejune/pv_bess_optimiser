# Coding Standards and Development Guidelines

Welcome to the ORS-BESS-Optimizer project! This guide will help you follow best practices and maintain code quality throughout the development process.

---

## Table of Contents
1. [Initial Setup](#initial-setup)
2. [Managing Dependencies](#managing-dependencies)
3. [Branch Naming Convention](#branch-naming-convention)
4. [Branching Workflow](#branching-workflow)
5. [Commit Message Convention](#commit-message-convention)
6. [Commit Squashing](#commit-squashing)
7. [Merge Request Process](#merge-request-process)
8. [Running Tests](#running-tests)
9. [Code Linting and Formatting](#code-linting-and-formatting)

---

## Initial Setup

Before you start development, ensure you have the development dependencies installed:

```bash
# Install the project with dev dependencies
pip install -e ".[dev]"
```

This will install:
- `pytest` and `pytest-cov` for testing
- `black` for code formatting
- `ruff` for linting
- `mypy` for type checking

---

## Managing Dependencies

When your feature requires external Python packages, follow this process to ensure dependencies are properly installed, documented, and tracked.

### Installing New Dependencies

#### 1. Install the Package for Development

Use `pip install` to add the package to your environment:

```bash
# Install a single package
pip install requests

# Install a specific version
pip install pandas==3.0.0

# Install with minimum version
pip install numpy>=2.0.0
```

#### 2. Add to pyproject.toml

The `pyproject.toml` file is the single source of truth for all project dependencies. Add your new dependency to the appropriate section:

##### Production Dependencies

For packages required to **run the application** (e.g., data processing, optimization, API clients), add to the `dependencies` list:

```toml
[project]
dependencies = [
    "pandas>=3.0.0",
    "numpy>=2.0.0",
    "pyomo>=6.5.0",
    "pydantic>=2.0.0",
    "python-dateutil>=2.8.0",
    "requests>=2.31.0",  # <- Add your new dependency here
]
```

##### Development Dependencies

For packages needed only during **development** (e.g., testing, linting, formatting), add to `dev` optional dependencies:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=9.0.0",
    "pytest-cov>=4.0.0",
    "black>=24.0.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
    "pytest-mock>=3.12.0",  # <- Add dev dependencies here
]
```

##### Machine Learning Dependencies

For ML-specific packages (e.g., forecasting, model training), add to `ml` optional dependencies:

```toml
ml = [
    "scikit-learn>=1.3.0",
    "scipy>=1.10.0",
    "tensorflow>=2.15.0",  # <- Add ML dependencies here
]
```

##### Solver Dependencies

For optimization solvers, add to `solvers` optional dependencies:

```toml
solvers = [
    "highspy>=1.5.0",
    "glpk>=0.5.0",
    "cbc>=2.10.0",  # <- Add solver dependencies here
]
```

##### Documentation Dependencies

For documentation generation tools, add to `docs` optional dependencies:

```toml
docs = [
    "sphinx>=7.0.0",
    "sphinx-rtd-theme>=2.0.0",
    "myst-parser>=2.0.0",  # <- Add doc dependencies here
]
```

#### 3. Reinstall the Project with New Dependencies

After updating `pyproject.toml`, reinstall the project to register the new dependencies:

```bash
# For production dependencies
pip install -e .

# For development dependencies
pip install -e ".[dev]"

# For multiple dependency groups
pip install -e ".[dev,ml,solvers]"

# Install all optional dependencies
pip install -e ".[all]"
```

### Version Pinning Guidelines

When specifying dependency versions, follow these conventions:

```toml
# ✅ Good: Minimum version with >= (allows patch updates)
"pandas>=3.0.0"
"numpy>=2.0.0"

# ✅ Good: Compatible release with ~= (allows patch updates, not minor)
"requests~=2.31.0"  # Allows 2.31.x but not 2.32.0

# ✅ Good: Exact version for critical stability
"tensorflow==2.15.0"

# ❌ Bad: No version constraint (unpredictable)
"scikit-learn"

# ❌ Bad: Upper bounds without reason (prevents updates)
"pandas>=2.0.0,<3.0.0"
```

**When to use each:**
- `>=` - Default choice for most dependencies
- `~=` - When you want patch updates only
- `==` - Only when exact version is critical (rare)

### Updating Documentation

After adding dependencies, update relevant documentation:

#### 1. Update README.md

If the dependency affects installation or usage, update the main README:

```markdown
## Dependencies

### Core Dependencies
- pandas >= 3.0.0 - Data manipulation and analysis
- numpy >= 2.0.0 - Numerical computing
- pyomo >= 6.5.0 - Optimization modeling
- requests >= 2.31.0 - HTTP library for API calls

### Optional Dependencies
- scikit-learn >= 1.3.0 - Machine learning algorithms
- highspy >= 1.5.0 - High-performance LP/MIP solver
```

#### 2. Update Feature Documentation

If your feature introduces a new dependency, document it in your merge request:

```markdown
## Dependencies Added

### requests >= 2.31.0
- **Purpose:** Fetch real-time energy prices from external API
- **Category:** Production dependency
- **Files using it:** `src/ors/data/price_fetcher.py`
- **Installation:** Included in base install (`pip install -e .`)
```

#### 3. Add Type Stubs (if needed)

If mypy complains about missing type information, add the package to mypy overrides:

```toml
[[tool.mypy.overrides]]
module = [
    "pandas.*",
    "pyomo.*",
    "requests.*",  # <- Add here if types are unavailable
]
ignore_missing_imports = true
```

### Complete Example Workflow

Let's say you need to add `httpx` for async HTTP requests:

```bash
# 1. Install the package for testing
pip install httpx

# 2. Test your code works
pytest tests/test_async_client.py

# 3. Edit pyproject.toml and add to dependencies
# Add line: "httpx>=0.27.0",

# 4. Reinstall project to register dependency
pip install -e ".[dev]"

# 5. Verify installation
pip show httpx

# 6. Update README.md with new dependency info

# 7. Commit with proper message
git add pyproject.toml README.md
git commit -m "[GRID-XX] Add httpx for async API requests

- Add httpx>=0.27.0 to production dependencies
- Update README with dependency information
- Implement async price fetcher using httpx"

# 8. Mention in merge request description
```

### Checking Dependencies

```bash
# List all installed packages
pip list

# Show specific package details
pip show pandas

# Check for outdated packages
pip list --outdated

# Generate requirements for reference (not used in this project)
pip freeze > requirements.txt  # For documentation only
```

### Common Pitfalls

❌ **Don't** install packages without adding to `pyproject.toml`:
```bash
pip install requests  # Works locally but breaks for others!
```

❌ **Don't** use `requirements.txt` - we use `pyproject.toml`:
```bash
# This project doesn't use requirements.txt
```

❌ **Don't** add test/dev tools to production dependencies:
```bash
# Bad - pytest should be in [project.optional-dependencies.dev]
dependencies = ["pytest>=9.0.0"]
```

✅ **Do** add to pyproject.toml and reinstall:
```bash
# 1. Edit pyproject.toml
# 2. Then run:
pip install -e ".[dev]"
```

✅ **Do** specify why you need a dependency in your commit/MR:
```markdown
## Dependencies Added
- httpx>=0.27.0: Required for async API calls to energy price service
```

---

## Branch Naming Convention

All branches **must** be prefixed with the Jira ticket code to maintain traceability between code changes and project requirements.

### Format
```
<JIRA-TICKET>-<brief-description>
```

### Examples
- `GRID-54-implement-battery-optimizer`
- `GRID-123-fix-energy-calculation`
- `GRID-87-add-unit-tests`

### Creating a Branch
```bash
# Create and checkout a new branch
git checkout -b GRID-54-your-feature-name

# Push the branch to remote
git push -u origin GRID-54-your-feature-name
```

---

## Branching Workflow

**IMPORTANT:** All feature branches **must** be created from the `dev` branch, NOT the `master` branch. The `master` branch contains production code and should only receive merges from `dev` after thorough team review.

### Development Flow

```
master (production)
  └── dev (development)
       ├── GRID-54-implement-battery-optimizer
       ├── GRID-123-fix-energy-calculation
       └── GRID-87-add-unit-tests
```

### Creating a Feature Branch

Always branch from `dev`:

```bash
# 1. Ensure you're on dev and have latest changes
git checkout dev
git pull origin dev

# 2. Create your feature branch from dev
git checkout -b GRID-54-your-feature-name

# 3. Push the branch to remote
git push -u origin GRID-54-your-feature-name
```

### Key Rules
- ✅ Feature branches → created from `dev`
- ✅ Merge requests → target `dev` branch
- ✅ `dev` → `master` merges → reviewed by whole team
- ❌ NEVER create feature branches from `master`
- ❌ NEVER merge feature branches directly to `master`

---

## Commit Message Convention

Every commit message **must** be prefixed with the Jira ticket code in square brackets to link commits to their respective tickets.

### Format
```
[JIRA-TICKET] Brief description of changes

Optional longer description explaining:
- What was changed
- Why it was changed
- Any relevant context
```

### Examples
```bash
# Good commit messages
git commit -m "[GRID-54] Add battery state of charge calculation"
git commit -m "[GRID-123] Fix energy price data parsing bug"
git commit -m "[GRID-87] Add unit tests for optimizer module"

# With detailed description
git commit -m "[GRID-54] Implement BESS optimization algorithm

- Add pyomo-based optimization model
- Integrate with price forecasting module
- Handle battery constraints and degradation"
```

### Bad Examples ❌
```bash
git commit -m "fixed bug"                    # Missing ticket prefix
git commit -m "GRID-54 added feature"        # Missing square brackets
git commit -m "[GRID-54]fixed typo"          # Missing space after prefix
```

---

## Commit Squashing

When your development work is complete and ready for review, you should **squash all commits** into one meaningful commit. This keeps the project history clean and makes it easier to understand what changes were made for each feature or bug fix.

### Why Squash Commits?

- Maintains a clean, readable git history
- Groups related changes together
- Makes it easier to revert features if needed
- Simplifies code review process

### Final Commit Format

Your squashed commit should have:
- A clear, descriptive title with the Jira ticket
- Bullet points summarizing what was implemented

```
[JIRA-TICKET] Clear title describing the overall feature/fix

- First major change or component
- Second major change or component
- Third major change or component
```

### Examples of Good Squashed Commits

```
[GRID-54] Implement BESS optimization algorithm

- Add pyomo-based linear programming model
- Integrate real-time price forecasting API
- Implement battery state-of-charge constraints
- Add degradation cost calculation
- Include comprehensive unit tests

[GRID-123] Fix energy price data parsing bug

- Correct timezone handling for price data
- Add validation for missing price values
- Update price interpolation logic
- Add integration tests for edge cases

[GRID-87] Add comprehensive test suite for optimizer module

- Add unit tests for battery constraint calculations
- Add integration tests for optimization workflow
- Add parametrized tests for various price scenarios
- Achieve 95% code coverage on optimizer module
```

### How to Squash Commits

#### Method 1: Interactive Rebase (Recommended)

```bash
# 1. Ensure your branch is up to date with dev
git checkout your-feature-branch
git fetch origin
git rebase origin/dev

# 2. Start interactive rebase (replace N with number of commits to squash)
# To find N, check how many commits ahead of dev you are
git log --oneline dev..HEAD  # Count the commits

# 3. Rebase the last N commits
git rebase -i HEAD~N

# Alternative: Rebase all commits since branching from dev
git rebase -i dev
```

In the interactive editor that opens:
- Keep the first commit as `pick`
- Change all other commits from `pick` to `squash` (or `s`)
- Save and close the editor

Example:
```
pick a1b2c3d [GRID-54] Initial implementation
squash e4f5g6h [GRID-54] Add tests
squash i7j8k9l [GRID-54] Fix linting issues
squash m0n1o2p [GRID-54] Address review comments
```

Then write your final commit message:
```
[GRID-54] Implement BESS optimization algorithm

- Add pyomo-based linear programming model
- Integrate real-time price forecasting API
- Implement battery state-of-charge constraints
- Add degradation cost calculation
- Include comprehensive unit tests
```

#### Method 2: Soft Reset (Alternative)

```bash
# 1. Ensure you're on your feature branch
git checkout your-feature-branch

# 2. Soft reset to dev (keeps all changes staged)
git reset --soft dev

# 3. Create a single new commit with all changes
git commit -m "[GRID-54] Implement BESS optimization algorithm

- Add pyomo-based linear programming model
- Integrate real-time price forecasting API
- Implement battery state-of-charge constraints
- Add degradation cost calculation
- Include comprehensive unit tests"
```

#### Force Push After Squashing

After squashing, you'll need to force push (your branch history has changed):

```bash
# Force push to update remote branch
git push --force-with-lease origin your-feature-branch
```

⚠️ **Warning:** Only force push to your own feature branches, never to shared branches like `dev` or `master`!

---

## Merge Request Process

When your feature is complete, tested, and squashed into a single commit, you're ready to create a merge request (MR) to merge your changes into the `dev` branch.

### Pre-Merge Request Checklist

Before creating your merge request, ensure:

```bash
# ✅ All tests pass
pytest

# ✅ Code is properly formatted
black --check .

# ✅ No linting issues
ruff check .

# ✅ Type checking passes
mypy src/

# ✅ Commits are squashed into one meaningful commit
git log --oneline dev..HEAD  # Should show only 1 commit

# ✅ Branch is up to date with dev
git checkout your-feature-branch
git rebase origin/dev
git push --force-with-lease origin your-feature-branch
```

### Creating a Merge Request in GitLab

1. **Navigate to GitLab:**
   - Go to your project repository in GitLab
   - Click on **"Merge Requests"** in the left sidebar
   - Click **"New merge request"** button

2. **Select Branches:**
   - **Source branch:** Your feature branch (e.g., `GRID-54-implement-battery-optimizer`)
   - **Target branch:** `dev` (NOT `master`)
   - Click **"Compare branches and continue"**

3. **Fill in Merge Request Details:**
   - **Title:** Use your commit message title (e.g., `[GRID-54] Implement BESS optimization algorithm`)
   - **Description:** Include:
     - Link to Jira ticket: `https://gridminds.atlassian.net/browse/GRID-54`
     - Summary of changes (can copy from commit message bullet points)
     - Any special testing instructions
     - Screenshots/examples if relevant

   Example description:
   ```markdown
   ## Jira Ticket
   https://gridminds.atlassian.net/browse/GRID-54

   ## Changes
   - Add pyomo-based linear programming model
   - Integrate real-time price forecasting API
   - Implement battery state-of-charge constraints
   - Add degradation cost calculation
   - Include comprehensive unit tests

   ## Testing
   - All unit tests pass: `pytest tests/test_optimizer.py`
   - Integration tests verified with sample data
   - Achieved 95% code coverage

   ## Notes
   - Required updating pyproject.toml with pyomo dependency
   ```

4. **Assign Reviewers:**
   - **Minimum 2 reviewers required**
   - Reviewers **must NOT** have been directly involved in developing the feature
   - Select team members with relevant expertise
   - Use the "Reviewers" field in GitLab

5. **Additional Settings:**
   - ✅ Check **"Delete source branch when merge request is accepted"**
   - ✅ Check **"Squash commits when merge request is accepted"** (if not already squashed)
   - Leave **"Merge when pipeline succeeds"** checked if CI/CD is configured

6. **Submit:**
   - Click **"Create merge request"**
   - Notify reviewers via your team communication channel

### Merge Request Review Guidelines

#### For Developers (MR Author):
- Respond promptly to reviewer comments
- Make requested changes in new commits (don't squash during review)
- Re-request review after making changes
- Be open to feedback and suggestions

#### For Reviewers:
- Review code within 24-48 hours
- Check for:
  - Code quality and adherence to standards
  - Proper test coverage
  - Logical correctness
  - Performance considerations
  - Security implications
- Use GitLab's comment features:
  - Add line-specific comments
  - Mark threads as resolved when addressed
  - Approve or request changes

### Merging Dev to Master

When merging `dev` into `master` (for production releases):

1. **Create Merge Request:**
   - Source: `dev`
   - Target: `master`
   - Title: `[RELEASE] Version X.Y.Z - Feature summary`

2. **Team Review Required:**
   - **ALL team members should review** before approval
   - At least one senior developer must approve
   - Schedule a team sync if necessary to discuss

3. **Approval Process:**
   - Ensure all CI/CD pipelines pass
   - Verify staging environment is stable
   - Get explicit approval from team lead or project manager
   - All discussions must be resolved

4. **Post-Merge:**
   - Tag the release: `git tag -a v1.0.0 -m "Release version 1.0.0"`
   - Update changelog
   - Notify stakeholders



## Running Tests

This project uses `pytest` for testing. Tests are located in the `tests/` directory.

### Run All Tests
```bash
# Run all tests in the project
pytest

# Run all tests with verbose output
pytest -v

# Run tests with coverage report
pytest --cov=src --cov-report=term-missing
```

### Run Specific Test Files
```bash
# Run a single test file
pytest tests/test_optimizer.py

# Run a specific test file with verbose output
pytest tests/test_optimizer.py -v
```

### Run Specific Test Functions
```bash
# Run a specific test function
pytest tests/test_optimizer.py::test_battery_charge

# Run tests matching a pattern
pytest -k "battery"

# Run tests by marker (unit, integration, slow)
pytest -m unit              # Only unit tests
pytest -m "not slow"        # Exclude slow tests
pytest -m integration       # Only integration tests
```

### Test with Coverage
```bash
# Generate coverage report
pytest --cov=src --cov-report=html

# View the HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Useful Pytest Options
```bash
# Stop on first failure
pytest -x

# Show local variables in tracebacks
pytest -l

# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Show print statements
pytest -s
```

---

## Code Linting and Formatting

This project uses three main tools for code quality:
1. **Black** - Code formatter
2. **Ruff** - Fast Python linter
3. **Mypy** - Static type checker

### Running from Terminal

#### Format Code with Black
```bash
# Format all Python files
black .

# Check what would be formatted (dry run)
black --check .

# Format specific file or directory
black src/ors/optimizer.py
black tests/
```

#### Lint Code with Ruff
```bash
# Lint all files
ruff check .

# Lint with auto-fix
ruff check --fix .

# Lint specific file or directory
ruff check src/ors/optimizer.py
ruff check tests/
```

#### Type Check with Mypy
```bash
# Type check all files
mypy src/

# Type check specific file
mypy src/ors/optimizer.py
```

#### Run All Quality Checks
```bash
# Run all checks in sequence
black --check . && ruff check . && mypy src/ && pytest
```

### IDE Setup

#### Visual Studio Code (VS Code)

1. **Install Extensions:**
   - Python (Microsoft)
   - Pylance (Microsoft)
   - Black Formatter (Microsoft)
   - Ruff (Charliermarsh)

2. **Configure Settings** (`.vscode/settings.json`):
```json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "100"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true,
    "source.fixAll": true
  },
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  }
}
```

#### PyCharm / IntelliJ IDEA

1. **Black Formatter:**
   - Go to: `Settings/Preferences` → `Tools` → `External Tools`
   - Add New Tool:
     - Name: `Black`
     - Program: `$PyInterpreterDirectory$/black`
     - Arguments: `$FilePath$`
     - Working directory: `$ProjectFileDir$`

2. **Ruff Linter:**
   - Go to: `Settings/Preferences` → `Tools` → `External Tools`
   - Add New Tool:
     - Name: `Ruff`
     - Program: `$PyInterpreterDirectory$/ruff`
     - Arguments: `check $FilePath$`
     - Working directory: `$ProjectFileDir$`

3. **Enable Pytest:**
   - Go to: `Settings/Preferences` → `Tools` → `Python Integrated Tools`
   - Set Default test runner to: `pytest`

4. **Mypy Type Checking:**
   - Install Mypy plugin from: `Settings/Preferences` → `Plugins`
   - Configure in: `Settings/Preferences` → `Editor` → `Inspections` → `Python`

#### Vim/Neovim

Add to your config (e.g., with `nvim-lspconfig` and `null-ls`):

```lua
-- Using null-ls.nvim
local null_ls = require("null-ls")
null_ls.setup({
  sources = {
    null_ls.builtins.formatting.black.with({
      extra_args = { "--line-length", "100" }
    }),
    null_ls.builtins.diagnostics.ruff,
    null_ls.builtins.diagnostics.mypy,
  },
})
```

### Pre-commit Hook (Optional but Recommended)

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Format code with black
black .

# Run linter
ruff check --fix .

# Add formatted files
git add -u

# Run tests
pytest

# If tests fail, abort commit
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

---

## Quick Reference

### Daily Workflow
```bash
# 1. Create feature branch from dev
git checkout dev
git pull origin dev
git checkout -b GRID-XX-feature-name

# 2. Make changes and format/lint
black .
ruff check --fix .

# 3. Run tests
pytest

# 4. Commit with proper prefix
git commit -m "[GRID-XX] Your descriptive message"

# 5. Push to remote
git push origin GRID-XX-feature-name
```

### Before Opening a Merge Request
```bash
# 1. Run full quality check
black --check . && ruff check . && mypy src/ && pytest --cov=src

# 2. Squash commits into one
git rebase -i dev

# 3. Update with latest dev
git rebase origin/dev

# 4. Force push
git push --force-with-lease origin GRID-XX-feature-name
```

---

## Configuration Details

All tool configurations are defined in `pyproject.toml`:
- **Black**: Line length 100, Python 3.10+
- **Ruff**: Line length 100, includes isort, pycodestyle, pyflakes, etc.
- **Mypy**: Python 3.10, strict equality checks enabled
- **Pytest**: Tests in `tests/`, markers for unit/integration/slow tests

---

## Getting Help

- Project Documentation: https://gridminds.atlassian.net/wiki/spaces/ab/overview
- Ask the team on WhatsApp. 

---

**Remember:** Consistent code style and proper testing make collaboration easier and catch bugs early. When in doubt, run the linters and tests! 🚀
