# How to Publish `thermoml_fair` to PyPI

This guide outlines the steps to build and publish new versions of the `thermoml_fair` package to TestPyPI (for testing) and the official PyPI (for release).

## Prerequisites

1.  **Python Environment**: Ensure you have a Python environment (e.g., your `micromamba cpu` environment) where the following packaging tools are installed:

    ```bash
    pip install build twine
    ```

2.  **PyPI Accounts**: You need accounts on [TestPyPI](https://test.pypi.org/) and [PyPI](https://pypi.org/).

3.  **API Tokens (Recommended)**:
    *   Generate API tokens for both TestPyPI and PyPI. When generating, scope the token to the specific project (`thermoml_fair`) if possible.
    *   Store these tokens securely.

## Publishing Steps

### 1. Update Version Number

*   The **single source of truth** for the version is in `pyproject.toml`.
*   Open `thermoml_fair/pyproject.toml`.
*   Update the `version` string under the `[project]` section (e.g., `version = "1.0.3"`).

    ```toml
    [project]
    name = "thermoml_fair"
    version = "1.0.3" # <-- UPDATE THIS
    # ...
    ```

### 2. Clean Previous Builds (Important!)

*   Before building a new version, it's crucial to remove any old package files from your `dist/` directory. This prevents `twine` from attempting to upload outdated versions.
*   In your terminal, navigate to the project root directory.
*   Delete the `dist/` folder or its contents. For example, in PowerShell:

    ```powershell
    Remove-Item -Recurse -Force dist
    ```

    Or in bash:

    ```bash
    rm -rf dist/
    ```

    If the `dist` directory doesn't exist, you can skip this step.

### 3. Build the Package

*   Open your terminal and ensure your Python environment with `build` and `twine` is active (e.g., `micromamba activate cpu`).
*   Navigate to the project root directory (`cd path/to/thermoml_fair`).
*   Run the build command:

    ```bash
    python -m build
    ```

    This will create a new `dist/` directory containing the source distribution (`.tar.gz`) and wheel (`.whl`) files for the current version.

### 4. Upload to TestPyPI (for testing)

This step allows you to verify the package before releasing it publicly.

**Method A: Using a `.pypirc` file (Recommended for convenience if it works)**

*   Create or edit the file `C:\\Users\\<YourUsername>\\.pypirc` (on Windows) or `~/.pypirc` (on macOS/Linux).
*   Add your TestPyPI token:

    ```ini
    [testpypi]
    username = __token__
    password = your_testpypi_api_token_here
    ```

*   Upload using `twine`:

    ```bash
    twine upload --repository testpypi dist/*
    ```

**Method B: Providing token on the command line (Used when `.pypirc` causes issues)**

*   If you have issues with `.pypirc` (like permission errors), ensure it's removed or renamed so `twine` doesn't try to use it.
*   Then, run:

    ```bash
    twine upload --repository testpypi --username __token__ --password YOUR_TESTPYPI_TOKEN_HERE dist/*
    ```

    Replace `YOUR_TESTPYPI_TOKEN_HERE` with your actual TestPyPI API token.
    **Note**: The `dist/*` wildcard will attempt to upload all files in the `dist` directory. By cleaning the directory in Step 2, you ensure only the current version's files are targeted.
    **Last time, removing a problematic `.pypirc` file and then using this command-line token method worked when permission issues were encountered.**

### 5. Test Installation from TestPyPI

*   **Create a new, clean virtual environment**:

    ```bash
    python -m venv my_test_env
    ```

*   **Activate the environment**:
    *   Windows PowerShell:

        ```powershell
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
        .\my_test_env\Scripts\Activate.ps1
        ```

        (You might need to run `Set-ExecutionPolicy` once per session if scripts are disabled.)
    *   Windows CMD: `.\my_test_env\Scripts\activate.bat`
    *   macOS/Linux: `source my_test_env/bin/activate`

*   **Uninstall any old versions** (important if re-testing in the same environment):

    ```bash
    pip uninstall thermoml-fair
    ```

*   **Install your package from TestPyPI**:
    To avoid issues with `pip`'s resolver or cache, especially if previous attempts failed:

    ```bash
    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --no-cache-dir thermoml-fair==<your_version>
    ```

    Replace `<your_version>` with the exact version you uploaded (e.g., `1.0.2`). Using `--no-cache-dir` and specifying the exact version helps prevent conflicts. This will install dependencies like `typer` from the main PyPI if they are not on TestPyPI.

*   **Verify installed packages** (optional, for debugging):

    ```bash
    pip list
    ```

*   **Test basic functionality**:

    ```bash
    python -c "import thermoml_fair; from importlib.metadata import version; print(version('thermoml_fair'))"
    thermoml-fair --version
    ```

    (The Python command uses `importlib.metadata` which is the modern way to get the version specified in `pyproject.toml`.)

### 6. Upload to PyPI (Official Release)

**Once you are confident the package works correctly after testing from TestPyPI:**

*   **Method A: Using a `.pypirc` file**
    *   Ensure your `C:\\Users\\<YourUsername>\\.pypirc` or `~/.pypirc` file has a section for PyPI:

        ```ini
        [pypi]
        username = __token__
        password = your_pypi_api_token_here
        ```

    *   Upload using `twine`:

        ```bash
        twine upload dist/*
        ```

        (Twine uploads to PyPI by default if `--repository` is not specified)

*   **Method B: Providing token on the command line**
    *   Run:

        ```bash
        twine upload --username __token__ --password YOUR_PYPI_TOKEN_HERE dist/*
        ```

        Replace `YOUR_PYPI_TOKEN_HERE` with your actual PyPI API token.

### 7. Verify on PyPI

*   Go to [https://pypi.org/project/thermoml-fair/](https://pypi.org/project/thermoml-fair/).
*   Check that the new version is listed and the description/metadata looks correct.
*   You can also try installing it in a clean environment:

    ```bash
    pip install thermoml-fair==<your_version>
    ```

## Important Considerations

*   **`.pypirc` Security**: Be cautious with storing tokens directly in `.pypirc` on shared systems. Consider environment variables or `keyring` for better security if needed, though API tokens are generally safer than passwords.
*   **Idempotency**: PyPI and TestPyPI do not allow re-uploading the same filename/version. If an upload fails midway, you might need to delete the partially uploaded release from the PyPI/TestPyPI website before trying again, or increment the version number if the build itself was flawed.
*   **Cleanliness**: Always ensure your working directory is clean (no uncommitted changes) before tagging a release and publishing.
*   **Tagging (Git)**: It's good practice to tag the commit you're releasing in Git:

    ```bash
    git tag v1.0.3
    git push origin v1.0.3
    ```

    Replace `v1.0.3` with the actual version.

