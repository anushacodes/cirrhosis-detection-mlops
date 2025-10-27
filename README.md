ml cirrhosis detection pipeline
==============================

An end-to-end machine learning pipeline for predicting the stage of liver cirrhosis using patient clinical data. 

Built with DVC for versioning, Docker, FastAPI and streamlit for containerization and deployment.



Project Organization
------------
    
    cirrhosis-detection-mlops/
    │
    ├── app/
    │   └── app.py                # FastAPI backend
    │   └── ui.py                # streamlit frontend

    │
    ├── src/
    │   ├── data.py               # Preprocessing and preprocessor creation
    │   ├── features.py           # Train/test split
    │   ├── model.py              # Training pipeline
    │   └── evaluate.py           # Model evaluation
    │
    ├── models/
    │   ├── processing.pkl        # Saved preprocessor
    │   └── model.json            # Trained XGBoost model
    │
    ├── data/
    │   ├── raw/                  # Raw dataset
    │   └── processed/            # Cleaned dataset
    │
    ├── app_ui.py                 # Streamlit UI
    ├── config.yaml               # Configuration
    ├── dvc.yaml                  # DVC pipeline definition
    └── main.py                   # Main orchestration script


--------

Things to try/add for future projects: experiment tracking with mlflow, ci/cd with github actions, push to docker hub, serverless deployment with aws lambda.

---------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
