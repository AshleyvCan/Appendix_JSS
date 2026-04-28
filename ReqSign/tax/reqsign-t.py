import pandas as pd
import time
import json
import re

from models.decoder_model import CausalLM

import argparse
from utils.extract_response import extract_tax_recom
from utils.preprocess import preprocessing
import os


def missing_label_number(LLM, messages, fi_available = None):
    prompt_final = "The response did not include the label number on the last line. Please provide the recommended number on the last line without explanation."
    if fi_available:
        prompt_final += f"Here are the definitions again: {fi_available}"
    messages.append({"role": "user", "content": prompt_final}) 

    messages, response = LLM.continue_conversation(messages)
    return messages, response

def incorrect_label_number(LLM, messages, fi_available = None):
    prompt_final = "The response did not include the label number associated with a category. Please provide the correct number on the last line without explanation."
    if fi_available:
        prompt_final += f"Here are the definitions again: {fi_available}"
    messages.append({"role": "user", "content": prompt_final}) 
    messages, response = LLM.continue_conversation(messages)
    return messages, response


def start_chat(LLM, system_prompt, user_prompt, fi_available = None): 

    # Start conversation with the LLM to categorize based on the dim
    messages, response = LLM.base_conversation(system_prompt, user_prompt, no_system=args.system_prompt)
    response =  response.rstrip()

    # If there is no label predicted on the last line, request the label again.
    if len(re.findall(r'\d+', response.strip(" \t").split('\n')[-1])) == 0:
        messages, response = missing_label_number(LLM, messages, fi_available)

    # In the case of categorizing for the 'requirement intention' dimension and if there is no valid predicted label on the last line, request the label again
    elif fi_available and len([int(x) for x in re.findall(r'\d+', response.split('\n')[-1]) if int(x) >=-1 and int(x) < 11]) == 0:
        messages, response = incorrect_label_number(LLM, messages, fi_available)

    return messages, response.strip(" \t")


def get_data_categories(LLM, prompts, dim, req):

    # Request the LLM to categorize based on the given dimension
    messages, response = start_chat(LLM, prompts['system_prompts'][dim], prompts['user_prompt'].format(user_story= req))

    # Filter out the predicted label
    response = response.rstrip()

    # If the final response still does not have the final prediction on the last line, check whether the model gave a final in another line (but only when one label is provided in the response)
    if len(re.findall(r'\d+', response.strip(" \t").split('\n')[-1])) == 0 and len(re.findall(r'\d+', response)) == 1 and [int(x) for x in re.findall(r'\d+', response)][0] in [0,1]:
        num = [int(x) for x in re.findall(r'\d+', response)][0]
    elif len([int(x) for x in re.findall(r'\d+', response.split('\n')[-1]) if int(x) in [0, 1]]) > 0:
        num = [int(x) for x in re.findall(r'\d+', response.split('\n')[-1]) if int(x) in [0, 1]][-1]
    else:
        num = -1

    return messages, response, num


def get_intent_category(LLM, prompts, fi_available, selected_patterns, req):

    # Add an option to not select an category
    if args.add_optional == 1:
        fi_available += "\n-1: None of these categories fit the requirement."

    # Given the available categories, select the corresponding examples for each category.    
    fi_examples = [prompts['examples'][str(i)] for i in list(set(selected_patterns['FI']))]
    fi_examples.sort(key=lambda s: int(re.search(r'(\d+)\s*$', s).group(1)))

    fi_examples = '\n\n'.join(fi_examples)

    # Request the LLM to categorize based on the given dimension
    messages, response = start_chat(LLM, prompts['system_prompts']['FI'].format(fi_dims = fi_available, examples=fi_examples),  prompts['user_prompt'].format(user_story= req), fi_available)

    response = response.rstrip()
    
    if len(re.findall(r'-?\d+', response.split('\n')[-1])) > 0:
        num = [int(x) for x in re.findall(r'-?\d+', response.split('\n')[-1]) if int(x) >=-1 and int(x) < 11][-1]
    else:
        num = -1

    return messages, response, num



def get_reranked_list(LLM, pattern_options, req, all_messages, recom):

    # Combine descriptions of UI Design patterns to be added to prompt
    pattern_names = [row['title'] for _, row in pattern_options.iterrows()]
    patterns_final = '\n\n'.join([f"[{row['ID']}] {row['title']}: \n'{row['description']}'" for _, row in pattern_options.iterrows()])
    
    # if there are suitable patterns available, ask the LLM to re-rank the patterns. Otherwise, return -1
    if len(pattern_options) >0:
        messages, response =  LLM.rerank(req, patterns_final, no_system=args.system_prompt)
    else:
        messages = '-1'
        response = '-1'

    all_messages['final'] = messages
    response = response.rstrip()
    recom['responses']['final'] = response

    # If the response of the LLM does not have any predicted label on the last line, request the ranked list again.
    
    if len([int(n) for n in re.findall(r'\[(\d+)\]', response.split('\n')[-1])]) == 0 and "-1" not in response.split('\n')[-1]: #and len([int(n) for n in re.findall(r'\[(\d+)\]', response)]) == 0
        messages.append({"role": "user", "content": "Your response did not present the answer in the correct format on the last line. Use the following format: [1] < [2] < .. Don't forget the '<' signs, and make sure to place the final answer on the last line without explanation."}) 
        messages, response =  LLM.continue_conversation(messages)
        all_messages['final_correction'] = messages

        recom['responses']['final_initial'] = recom['responses']['final']
        response = response.rstrip()
        recom['responses']['final'] = response
    return all_messages, recom
    


def get_recom_taxonomy(LLM, prompts, req, selected_patterns):
    recom = {'responses': {}, "pred": {}}
    all_messages = {}
    
    # for the first two dimensions (Data insertion, Data retrieval), prompt the LLM to categorize the requirement.
    for dim in ['DI', 'DR']:
        messages, response, num = get_data_categories(LLM, prompts, dim, req)
        selected_patterns = selected_patterns[(selected_patterns[dim] == num)]
        all_messages[dim] = messages
        recom['responses'][dim] = response
        recom['pred'][dim] = num

    # for the "functional intention" dimension, prompt the LLM twice to categorize the requirement.
    for i in range(2):
        dim = 'FI'
        
        # Based on dimension DI and DR, select the UI Design pattern subset still available and their relating requirement intention label
        if i == 0: 
            fi_available =[prompts['fi_dims'][str(i)] for i in list(set(selected_patterns['FI']))]

        # In addition, don't select a category that is previously chosen
        else: 
            fi_available = [prompts['fi_dims'][str(i)] for i in list(set(selected_patterns[selected_patterns['FI'] != num]['FI']))]
        fi_available.sort(key=lambda s: int(re.match(r'\s*(\d+)', s).group(1)))
        fi_available = '\n\n'.join(fi_available)

        # Prompt the LLM
        messages, response, num = get_intent_category(LLM, prompts, fi_available, selected_patterns[selected_patterns['FI'] != num], req)
        all_messages[dim + str(i)] = messages
        recom['responses'][dim + str(i)] = response
        recom['pred'][dim + str(i)] = num

    # Re-rank available patterns based on the three dimensions
    pattern_options = selected_patterns[(selected_patterns['FI'] == recom['pred']['FI0']) | (selected_patterns['FI'] == recom['pred']['FI1'])]
    all_messages, recom = get_reranked_list(LLM, pattern_options, req, all_messages, recom)

    # Uncertainty estimation
    patterns_final = '\n\n'.join([f"[{row['ID']}] {row['title']}: \n'{row['description']}'" for _, row in pattern_options.iterrows()])
    probs = LLM.estimate_confidence(patterns_final, req, args.system_prompt)
    recom['pred']['probs'] = probs

    return recom, all_messages, probs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--why", "-w",type=int,
                        help="Use why part of the User Story")

    parser.add_argument("--model", "-m",type=str,
                        help="Model for taxonomy")
    parser.add_argument("--token", "-t",type=str,
                        help="TOKEN")
    parser.add_argument("--dataset", "-d",type=str,
                        help="Dataset")
    parser.add_argument("--add_optional", "-ao",type=int,
                        help="Add 'non' option to classification")
    parser.add_argument("--system_prompt", "-sp",type=int,
                        help="Whether to have a system prompt or only user prompt")
    global args
    args = parser.parse_args()

    

    # Import the taxonomy information and the user story dataset
    df_patterns = pd.read_excel('data/taxonomy_patterns.xlsx')
    df_userstories = pd.read_excel(args.dataset, sheet_name="Interview 2")

    if args.dataset.split('/')[0] == "students":
        group = args.dataset.split('/')[1]
    else:
        group = args.dataset[:3]

    # Identify user stories and remove why part
    proc_requirements, requirements = preprocessing(df_userstories, args.why)


    output_dir = f"{group}_{args.model[:4]}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Import all prompts
    prompts = {'system_prompts': {}}
    with open('models/prompts/prompt_dr.txt', "r") as outfile:
        prompts['system_prompts']['DR'] = outfile.read()
    with open('models/prompts/prompt_di.txt', "r") as outfile:
        prompts['system_prompts']['DI'] = outfile.read()
    with open('models/prompts/prompt_fi.txt', "r") as outfile:
        prompts['system_prompts']['FI'] = outfile.read()
    with open('models/prompts/user_prompt.txt', "r") as outfile:
        prompts['user_prompt'] = outfile.read()
    with open('models/prompts/user_prompt_fi.txt', "r") as outfile:
        prompts['user_prompt_fi'] = outfile.read()
    with open('models/prompts/fi_dim.json', 'r') as f:
        prompts['fi_dims'] = json.load(f)
    with open('models/prompts/fi_dim_examples.json', 'r') as f:
        prompts['examples'] = json.load(f)
 
    recoms = {}
    all_messages = {}
    rows_recom = []

    LLM = CausalLM(args.model, args.token)
    print(args.model)
    all_probs = []

    # For each requirement in the dataset: get most relevant UI Design patterns. 
    for i in range(len(proc_requirements)): 

        user_story = requirements[i]
        proc_us = proc_requirements[i]
        # Prompt the LLM to categorize each requirement, based on the three dimensions and subsequently rank the UI Design patterns to the category
        # recom: the prediction labels
        # messages: for each dimension, the conversation history
        # 
        recom, messages, probs = get_recom_taxonomy(LLM, prompts, proc_us, df_patterns)

        recoms[user_story] = recom
        all_messages[user_story] = messages
        all_probs.append(probs)
        rows_recom.append( {"user_story": user_story} | extract_tax_recom(recom, df_patterns))
        
        # Update the prediction and messages file
        with open(output_dir+"/responses_fi.json", "w", encoding="utf-8") as f:
            json.dump(recoms, f, indent=4)

        with open(output_dir+"/messages_fi.json", "w", encoding="utf-8") as f:
            json.dump(all_messages, f, indent=4)

    df = pd.DataFrame(rows_recom)
    
    # Add column to df with certainty estimations
    df["prob_0"] = [p[0]['0'] for p in all_probs]
    df["prob_1"] = [p[0]['1'] for p in all_probs]
    df["logprob_0"] = [p[1]['0'] for p in all_probs]
    df["logprob_1"] = [p[1]['1'] for p in all_probs]
    
    df.to_excel(output_dir+f"/recommendations_tax_{group}_{args.model[:4]}.xlsx")

if __name__ == "__main__":
    main()

