
import pandas as pd
import re


def preprocessing(df_userstories, arg_why = 0):
    # Identify user stories
    if df_userstories["Epic/Story"].isin(["S", "Story"]).any():
        df_userstories = df_userstories[df_userstories["Epic/Story"].isin(["S", "Story"])]

    requirements = df_userstories['Requirement'].to_list()

    # Removing why part
    if arg_why == 0:
        proc_requirements = [re.split(r"\b(?:so that|because)\b|,\s*so\b", req, maxsplit=1)[0].strip().rstrip(",") for req in requirements]
    else:
        proc_requirements = [req for req in requirements]

    return proc_requirements, requirements
