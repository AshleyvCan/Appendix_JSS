
from models.decoder_model import CausalLM
import pandas as pd
import json
import re
import argparse

from utils.extract_response import extract_base_recom
import time

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", "-m",type=str,
                        help="Model for taxonomy")
    parser.add_argument("--token", "-t",type=str,
                        help="TOKEN")
    parser.add_argument("--system_prompt", "-sp",type=int,
                        help="Whether to have a system prompt or only user prompt")
    parser.add_argument("--dataset", "-d",type=str,
                        help="Dataset")
    args = parser.parse_args()


    if args.dataset.split('/')[0] == "students":
        group = args.dataset.split('/')[1]
    else:
        group = args.dataset[:3]
    
    # Determine input and output folder
    input_dir = f"{group}_e5"

    # For each requirement, get the candidate patterns 
    with open(input_dir+f"/retrieval_results_e5_{group}.json", "r") as f:
        retrieval_results = json.load(f)

    LLM = CausalLM(args.model, args.token)


    # For each requirement, rank the candidate patterns
    all_messages = {}
    all_probs = []
    for i, req in enumerate(retrieval_results.keys()):
        patterns_text = '\n\n'.join([f'[{str(i)}] {text}' for i, text in enumerate(retrieval_results[req]["pattern_desc"])]) #
        proc_req = re.split(r"\b(?:so that|because)\b|,\s*so\b", req, maxsplit=1)[0].strip().rstrip(",")

        messages, response =  LLM.rerank(proc_req, patterns_text, args.system_prompt)
        retrieval_results[req]['rank'] = response

        probs = LLM.estimate_confidence(patterns_text, proc_req, args.system_prompt)
        all_probs.append(probs)
        all_messages[req] = messages

    with open(input_dir+f"/retrieval_results_e5_{group}_{args.model[:4]}.json", "w") as f:
        json.dump(retrieval_results, f, indent=4)    
    
    with open(input_dir+f"/messages_e5_{group}_{args.model[:4]}.json", "w", encoding="utf-8") as f:
        json.dump(all_messages, f, ensure_ascii=False, indent=4)

    # Extract the predictions from the repsonses
    rows_base, _ = extract_base_recom(retrieval_results)
    df = pd.DataFrame(rows_base)

    # Add columns to the df for the certainty estimation
    df["prob_0"] = [p[0]['0'] for p in all_probs]
    df["prob_1"] = [p[0]['1'] for p in all_probs]
    df["logprob_0"] = [p[1]['0'] for p in all_probs]
    df["logprob_1"] = [p[1]['1'] for p in all_probs]


    df.to_excel(input_dir+f"/recommendations_base_{group}_{args.model[:4]}.xlsx")


if __name__ == "__main__":
    main()
