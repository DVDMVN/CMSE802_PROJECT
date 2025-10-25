# **Predicting Kickstarter Success from Launch Information**

This project leverages historical Kickstarter campaign data to predict whether a project will succeed or fail based on launch-time information such as topic, category, description, funding goal, and campaign duration. Beyond building accurate models, we aim to understand how predictive patterns vary across creative domains and whether specialized models outperform general ones. The overarching goal is to generate interpretable, data-driven recommendations that help creators optimize their campaign strategies.

### Objectives

1. Compare general vs. category-specific models to evaluate whether predictors of success differ meaningfully across creative domain
2. Identify key success factors within each domain to provide actionable insights for creators.

### Folder structure

- 📁 `data/raw`, `data/processed`: folders for holding both the raw and post processing versions of the dataset. Since the dataset is so massive, these will be populated locally only via `src/download_data.py`
- 📁 `notebooks`: folder for holding exploratory notebooks for IDA and EDA. For a comprehensive overview of decision making, insights, and results, see these notebook(s).
- 📁 `src`: folder for holding all main pipeline stages and finalized methods for training, testing, and evaluating models, as well as producing results and figures.
    - 📄 `config.py` - Holds project wide configuration and paths
    - 📄 `download_data.py` - Script to fetch and organize the Webrobots Kickstarter dumps into `data/raw`
    - 📄 `preprocessing.py` – Script for cleaning, merging, feature engineering, etc. and saving processed datasets into `data/processed`.
    - 📄 `modeling.py` – Script for running the final machine learning pipeline. Results are saved in the results folder.
    - 📄 `utils.py` – Helper functions for usage across many files.
- 📁 `results`: Stores the generated results from experimentation.
    - 📁 `figures` – Generated plots and visualizations.
    - 📁 `models` – Serialized trained models for checkpointing.
- 📁 `reports`: Folder for presentation slide-decks or report write-ups.
- 📄 `.gitignore`: Git tracking configuration.
- 📄 `README.md`: Project description.
- ⚙️ `pyproject.toml`: Python project configuration file, includes the list of all python dependencies.

### Setup and running

1. Clone the repository:
   ```bash
   git clone https://github.com/DVDMVN/CMSE802_PROJECT.git
   ```

2. Create a virtual environment and install all dependencies listed from `pyproject.toml`
- This project is built using [`uv`](https://docs.astral.sh/uv/) package manager. Run `uv sync` in the root directory. This is the `uv` equivalent to:
    - python -m venv .venv
    - pip install -r requirements.txt
- After `uv sync` either continue by using `uv run <file_path>` or activate the virtual environment in the traditional way by running `.venv/bin/activate` and continue by using `python <file_path>`.

3. Run `src/*.py` files to run parts of the Machine Learning pipeline:
    - Run `download_data.py` via CLI to download the kickstarter datasets. Warning, this will be very large, and time consuming.
    - Run `preprocessing.py` via CLI to read raw CSVs, preprocess them, and save a processed parquet to `data/processed`.
    - Run `modeling.py` via CLI to train and evaluate models and write accuracy results into the `results/` folder.
