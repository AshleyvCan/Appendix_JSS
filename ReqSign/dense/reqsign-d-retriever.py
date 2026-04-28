from models.e5 import get_similarity_e5 
import argparse

import pandas as pd
import json
import re
import time
import os
from utils.preprocess import preprocessing

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", "-d",type=str,
                        help="Dataset")
    parser.add_argument("--why", "-w",type=int,
                    help="Use why part of the User Story")
    args = parser.parse_args()

    # Import UI Design patterns
    df_patterns = pd.read_excel('data/taxonomy_patterns.xlsx')

    # Import requirements
    df_userstories = pd.read_excel(args.dataset, sheet_name= "Interview 2") 
    patterns = [f"{row['title']}: \n'{row['description']}'" for _, row in df_patterns.iterrows()]
    
    if args.dataset.split('/')[0] == "students":
        group = args.dataset.split('/')[1]
    else:
        group = args.dataset[:3]

    # Identify user stories and remove why part
    proc_requirements, requirements = preprocessing(df_userstories, args.why)

    # Create output folder
    output_dir = f"{group}_e5"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate similarity
    scores = get_similarity_e5(proc_requirements, patterns)
    
    patterns_titles = [f"{row['title']}" for _, row in df_patterns.iterrows()]


    # Get for each requirement, the top k patterns
    k = 8
    retrieval_results = {}
    for i in range(len(scores)):
        patterns_sorted = [x for _, x in sorted(zip(scores[i], patterns_titles), reverse=True)]
        patterns_sorted2 = [x for _, x in sorted(zip(scores[i], patterns), reverse=True)]

        retrieval_results[requirements[i]] = {"pattern_name": patterns_sorted[:k], "pattern_desc": patterns_sorted2[:k]}

    with open(output_dir+f"/retrieval_results_e5_{group}.json", "w") as f:
        json.dump(retrieval_results, f, indent=4)




if __name__ == "__main__":
    main()

