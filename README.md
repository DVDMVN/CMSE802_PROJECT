# **Predicting Kickstarter Success from Launch Information**

Utilizing past Kickstarter project data to predict project success or failure using launch information features such as topic, category, description, fundraising goal, and timeframe. Our goal is to see whether we can build accurate predictive models and identify the key factor that influence outcomes to provide data-driven insights to future creators on the platform looking to optimize their campaign strategies.

### Objectives

1. Development of accurate models to predict campaign outcomes.
2. Extraction of actionable, data-driven insights for creators to utilize as recommendation.
3. Evaluate the role of multi-modal data in making better predictions.

### Folder structure

- 📁 `data/raw`, `data/processed`: folders for holding both the raw and post processing versions of the dataset. Since the dataset is so massive, these will be populated locally only via `src/download_data.py`
- 📁 `notebooks`: folder for holding exploratory notebooks for IDA and EDA
- 📁 `src`: folder for holding all main pipeline stages and finalized methods for training, testing, and evaluating models, as well as producing results and figures.
    - 📄 `download_data.py` - Script to fetch and organize the Webrobots Kickstarter dumps into `data/raw`
    - 📄 `preprocessing.py` – Function for cleaning, merging, feature engineering, etc. and saving processed datasets into `data/processed`.
    - 📄 `models.py` – Model definitions with hyperparameters and routine specifics.
    - 📄 `evaluate.py` – Evaluation pipeline for training testing and computing metrics for models.
    - 📄 `visualizations.py` – Plotting utilities for figures, such as feature importance, or other data visualizations.
    - 📄 `utils.py` – Helper functions for usage across many files.
- 📁 `results`: Stores the generated results from experimentation.
    - 📁 `figures` – Generated plots and visualizations.
    - 📁 `models` – Serialized trained models for checkpointing.
- 📁 `reports`: Folder for presentation slide-decks or report write-ups.
- 📄 `.gitignore`: Git tracking configuration.
- 📄 `README.md`: Project description.
- ⚙️ `requirements.txt`: List of all python dependencies

### Setup and running

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/kickstarter-success-prediction.git
   ```

2. Create a virtual environment and install all dependencies listed from `requirements.txt`

3. Run `download_data.py` via CLI to download the kickstarter datasets. Warning, this will be very large, and time consuming.

4. Run the other individual pipeline stages via CLI:
- `preprocessing.py` to create the processed versions of the raw datasets
- `evaluate.py` to train test and evaluate models defined in `models.py`
- `visualizations.py` to create visualizations for the results