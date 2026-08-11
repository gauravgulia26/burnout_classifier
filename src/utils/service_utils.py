import dagshub


def init_dagshub() -> str:
    dagshub.init(repo_owner="grvgulia007", repo_name="burnout_classifier", mlflow=True)
