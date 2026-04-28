# Online Appendix of the paper "Retrieving UI Design Patterns from Requirements: Automated Approaches and Experimentation"

In this paper, we propose ReqSign, an automated pipeline for pattern retrieval based on the requirements of a specific system.
The appendix contains two main folders: ReqSign, Supplementary material 

## Folder: ReqSign
This folder contains the implementation of the automated pipeline ReqSign. 
The post-processing stage is embedded within each file separately (but contains the same function).

To run ReqSign-D:
```
python3.9 -m dense.reqsign-d-retriever -w 0 -d [dataset] 
python3.9 -m dense.reqsign-d-reranker -m [model_name] -t [TOKEN]  -sp 1 -d [dataset] 
```

To run ReqSign-T:
```
python3.9 -m tax.reqsign-t -w 0 -m [model_name] -t [TOKEN] -d [dataset] -ao 1 -sp 1
```

An access token can be generated via: https://huggingface.co/docs/hub/security-tokens

### subFolder: data
This subfolder contains the taxonomy and an example requirements file (which can be used to run the pipeline).

### subFolder: dense
This subfolder contains the scripts to run ReqSign-D.

### subFolder: models
The models used in both versions of the pipeline are presented in this subfolder. Also, the prompts of the decoder-only models can be found here.

### subFolder: tax
This subfolder contains the scripts to run ReqSign-T.

### subFolder: utils
To run ReqSign, this folder focuses on the functionalities to extract the predictions from the model's responses.



## Folder: Supplementary material 
This folder consists of subfolders: RQ1, RQ2 and RQ3.


### subFolder: RQ1
This subfolder contains the following files:
- cohen_kappa_scores.xlsx: present the raw Cohen Kappa scores between the first author and the other tagger.
- ending_conditions.xlsx: contains the formulated ending conditions for the creation of the taxonomy
- taxonomy_patterns.xlsx: contains the taxonomy, where each pattern name is presented with the chosen characteristics.
- exclusion_criteria.xlsx: consists of the defined exclusion criteria for the patterns.

### subFolder: RQ2
The following files are present in this subfolder:
- demonstration_examples.xlsx: presents the demonstration examples used in the prompts for ReqSign-T. The table presents also the original user story and/or whether the examples was manually created.
- model_selection.ipynb: presents a script to retrieve the HuggingFace OpenLLM Leaderboard. To replicate the results, make sure you only select models with a submission date before 17-12-2025.

### subFolder: RQ3
This subfolder contains the following files:
- metrics_scores.xlsx: contains the raw metric scores to generate the boxplots of Figure 7.
- test_results.xlsx: presents the test results of each metric tested.
- histogram.xlsx: shows a table with the frequency values presented in Figure 8.
- common_categories.xlsx: presents the ratio and frequency scores to create Figure 10 & 12.
- thematic_analysis.xlsx: contains the frequency of occurrence of each mentioned theme in section 8.2.3.
- survey_questions.xlsx: presents the questions asked in the survey.
