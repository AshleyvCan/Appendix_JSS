import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import re


def extract_base_recom(file_base):
    recom_base = {}
    rows_base = []
    top_k = 3

    for k,v in file_base.items():
        recom_labels =  [int(n) for n in re.findall(r'\[(\d+)\]', v['rank'])]

        recom = [v['pattern_name'][n] for n in recom_labels if n < len(v['pattern_name'])]
        recom_base[k] = recom

        rows_base.append({'user_story': k} | {f'rank_{str(i+1)}': p for i, p in enumerate(recom[:top_k])} | {f'rank_ID_{str(i+1)}': p for i, p in enumerate(recom_labels[:top_k])}) #| {"certainty":v['pred']['certainty']
    return rows_base, recom_base


def extract_tax_recom(responses, df_patterns):
    top_k = 3   

    # Identify the prediction labels in the response
    recom_labels =  [int(n) for n in re.findall(r'\[(\d+)\]', responses['responses']['final'].split('\n')[-1])][:top_k]

    # if there were no suitable patterns available, recom_labels should have -1. In that case, there is no recommendation for this requirement
    # Otherwise: link the IDs to the names/titles of the patterns.
    if -1 in recom_labels or -1 in [int(x) for x in re.findall(r'-?\d+', responses['responses']['final'].split('\n')[-1])]:
        recom = [-1] * top_k
        recom_labels = [-1] * top_k
    else:
        recom = [df_patterns[df_patterns['ID'] == n]['title'].values[0] for n in recom_labels if n in df_patterns['ID']]

    return {f'rank_{str(i+1)}': p for i, p in enumerate(recom)} | {f'rank_ID_{str(i+1)}': p for i, p in enumerate(recom_labels)}
