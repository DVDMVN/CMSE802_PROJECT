# **Predicting Kickstarter Success from Launch Information**

This project seeks to investigate whether certain Kickstarter categories have systematic advantages in getting funded, and whether category-specific models outperform a general category-agnostic model. By comparing these approaches, we can determine if the predictive signals of success are universal across categories or if each category requires tailored modeling to capture its unique funding dynamics.

### Objectives

1. Do different categories of Kickstarter campaigns have different predictive patterns for success?
2. Are models trained within specific categories more predictive within their category than models that are trained generally?

### Folder structure

- 📁 `data/raw`, `data/processed`: folders for holding both the raw and post processing versions of the dataset. Since the dataset is so massive, these will be populated locally only via `src/download_data.py`
- 📁 `notebooks`: folder for holding exploratory notebook for full data science process. From preprocessing, IDA, EDA to modeling and optimization. For a comprehensive overview of decision making, insights, and results, see this notebook.
- 📁 `src`: folder for holding all main pipeline stages and finalized methods for training, testing, and evaluating models, as well as producing results and figures.
    - 📄 `config.py` - Holds project wide configuration and paths
    - 📄 `download_data.py` - Script to fetch and organize the Webrobots Kickstarter dumps into `data/raw`
    - 📄 `preprocessing.py` – Script for cleaning, merging, feature engineering, etc. and saving processed datasets into `data/processed`.
    - 📄 `modeling.py` – Script for running the final machine learning pipeline. Results are saved in the results folder.
    - 📄 `utils.py` – Helper functions for usage across many files.
- 📁 `results`: Stores the generated results from experimentation. Main results will be contained within this folder level.
    - 📁 `figures` – Generated plots and visualizations.
    - 📁 `optimization_trials` – hyperparameter trialing history to extract and retain best parameters for each model.
- 📁 `reports`: Folder for presentation slide-deck(s) or written report write-ups.
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
    - Run `download_data.py` via CLI to download the kickstarter datasets. This will write to `data/raw/`. Warning, this will be very large, and time consuming.
    - Run `preprocessing.py` via CLI to read raw CSVs, preprocess them, and save a processed parquet to `data/processed/`.
    - Run `modeling.py` or `modeling.py default` via CLI to train and evaluate base models. Results will be saved to `results/`
    - Run `optimization.py` via CLI to perform hyperparameter optimization for various models. Results will be saved to `results/optimization_trials/`. Warning, this may take a long computation time (roughly 15 - 30 minutes).
    - Run `modeling.py evaluate_optimized` via CLI to evaluate optimized models using best hyperparameters found during optimization. Results will be saved to `results/`. This requires previously running `optimization.py` to attain best hyperparameters.
        - The results include overall and category specific performance metrics on the best tuned XGBoost model and ensemble of tuned category specific XGBoost models.

To reproduce the full analysis, run the files in the order specified above.