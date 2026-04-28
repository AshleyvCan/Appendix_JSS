import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import torch.nn.functional as F
import numpy as np
import re
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

class CausalLM:

    def __init__(self,model_name, TOKEN = None):

        if TOKEN:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token = TOKEN)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map="auto",
                token = TOKEN
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map="auto",
            )
        
        self.base_rerank_system_prompt = self.import_prompts("models/prompts/prompt_rerank.txt")
        self.conf_prompt = self.import_prompts('models/prompts/prompt_certainty.txt')
    
    def import_prompts(self, file):
        with open(file, "r") as outfile:
            return outfile.read()
        
    def send_message(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=1000,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0, input_ids.shape[-1]:]
        text =self.tokenizer.decode(generated, skip_special_tokens=True)

        return text


    def start_messages_reranker(self, requirement, patterns_text, no_system = 0):
      
        system_message = {
            "role": "system",
            "content": self.base_rerank_system_prompt}

        user_message = {
            "role": "user",
            "content": f": This is the requirement: {requirement}. \n These are the patterns: \n {patterns_text}"}
        
        if no_system == 1:
            messages = [{"role":"user", "content": system_message["content"]+"\n\n"+user_message["content"]}]
        else:
            messages = [system_message, user_message]
        return messages
    

    def rerank(self,requirement, patterns_text, no_system = 0):
        messages = self.start_messages_reranker(requirement, patterns_text, no_system)
        response = self.send_message(messages)


        messages.append({"role": "assistant",
            "content":response})

        return messages, response


    def base_conversation(self,system_prompt, user_prompt, no_system = 0):
        
        system_message = {
            "role": "system",
            "content": system_prompt}

        user_message = {
            "role": "user",
            "content": user_prompt}

        if no_system == 1:
            messages = [{"role":"user", "content": system_message["content"]+"\n\n"+user_message["content"]}]
        else:
            messages = [system_message, user_message]
        response = self.send_message(messages)
        messages.append({ "role": "assistant",
            "content":response})

        return messages, response


    def continue_conversation(self, messages):
        response = self.send_message(messages)
        messages.append({"role": "assistant",
            "content":response})

        return messages, response



    def completion_logprob_chat(self, message, next_token, device="cuda"):

        # Tokenize the prompt and completion 
        prompt = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(device) 
        
        completion_tokens = self.tokenizer(next_token, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
        tokens = torch.cat([prompt_tokens, completion_tokens], dim=1)
        
        # Retrieving the logits of the outputs
        prompt_len = prompt_tokens.shape[-1]    
        completion_len = completion_tokens.shape[-1]   
        with torch.no_grad():   
            outputs = self.model(tokens) 
            logits = outputs.logits
        
        # Identify the ID of the completion token
        tok_id = tokens[0, -1].item()

        # Identify the logits of the completion token
        past_tok = prompt_len + completion_len - 2
        token_logit = logits[0, past_tok, :]

        # Convert logits to log-probabilities
        token_log_probs = torch.nn.functional.log_softmax(token_logit, dim=-1)

        # Identify the log-probability of the final token
        log_token_prob = token_log_probs[tok_id].item()

        return log_token_prob


    def get_estimates(self, system_prompt, user_prompt, no_system, device="cuda"):
        system_message = {
            "role": "system",
            "content": system_prompt}

        user_message = {
            "role": "user",
            "content": user_prompt}
        
        if no_system == 1:
            messages = [{"role":"user", "content": system_message["content"]+"\n\n"+user_message["content"]}]
        else:
            messages = [system_message, user_message]

        # Get next token probability
        lp1 = self.completion_logprob_chat(messages, next_token="1", device=device)
        lp0 = self.completion_logprob_chat(messages, next_token="0", device=device)

        # Normalize
        lps = torch.tensor([lp1, lp0], device=device)
        ps = torch.softmax(lps, dim=0).tolist()
        return {"1": ps[0], "0": ps[1]}, {"1": lp1, "0": lp0}

    def estimate_confidence(self, patterns_final, req, no_system):
        return self.get_estimates(self.conf_prompt,f"Here is the requirement: {req}\nHere are the UI Design patterns:\n{patterns_final}", no_system)
