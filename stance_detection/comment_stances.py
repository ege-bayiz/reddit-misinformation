import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import datetime as dt
import warnings
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import (LoraConfig, PeftConfig, AutoPeftModelForCausalLM, PeftModel)
import os
import shutil

## Loading the datasets
Tier_1  = pd.read_pickle('stance_detection/curated_datasets/Tier_1.pickle')
Tier_2  = pd.read_pickle('stance_detection/curated_datasets/Tier_2.pickle')
# Removing any shared comments from politics (I think we can keep politics comments by the way and random sample the user to get what we need)
Tier_1=Tier_1[Tier_1['sub']!='politics']
# Min number of words in body (3 right now)
Tier_1=Tier_1[Tier_1['body'].apply(lambda x: len(x.split()))>2]
# Years that you want your tier in
Year=[2016]
Tier_1=Tier_1[Tier_1['Y'].isin(Year)]

# Some more filters on tiers if needed
Tier_1_10plus = Tier_1['author'].value_counts()
Tier_1_10plus = Tier_1['author'].value_counts().reset_index()
Tier_1_10plus = Tier_1_10plus[Tier_1_10plus['count']>=10]
Tier_1  = Tier_1.loc[Tier_1['author'].isin(Tier_1_10plus['author'].to_list())]


data = Tier_1

## Loading the models
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

print(f'running on cuda device {torch.cuda.current_device()}')

### Loading the Llama2 7b - chat model
base_model_name = "NousResearch/Llama-2-7b-chat-hf"

llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map='sequential'
    #max_memory=max_memory,
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

print(base_model.hf_device_map)

lora_id = "stance-detection/trained_lora/llama-2-7b-reddit_stance_det_lora_2"
config = PeftConfig(lora_id)
inference_model = PeftModel.from_pretrained(base_model, lora_id)



## Label generation loop
text_gen = pipeline(task="text-generation", model=inference_model, tokenizer=llama_tokenizer, max_new_tokens=8)
query_head = "You are a helpful, respectful, and honest assistant that detects the stance of a comment with respect to its parent. Stance detection is the process of determining whether the author of a comment is in support of or against a given parent. You are provided with:\n post: the text you that is the root of discussion.\n parent:  the text which the comment is a reply towards.\n comment: text that you identify the stance from.\n\nYou will return the stance of the comment against the parent. Only return the stance against the parent and not the original post. Always answer from the possible options given below: \n support: The comment has a positive or supportive attitude towards the post, either explicitly or implicitly. \n against: The comment opposes or criticizes the post, either explicitly or implicitly. \n none: The comment is neutral or does not have a stance towards the post. \n unsure: It is not possible to make a decision based on the information at hand."

data['stance'] = 'unsure'
# Define the save path and backup path
save_path = 'stance_detection/labeled_no_politics/2016/Tier_1.pickle'
backup_path = 'stance_detection/labeled_no_politics/2016/Tier_1_backup.pickle'

ii = 0
# Iterate over each row and apply the placeholder function
for index, row in tqdm(data.iterrows()):
    query = f"<SYS> {query_head} </SYS>" + "\n\n" + "post: " + row['submission_body'] + "\n" + "parent: " + row['submission_body'] + "\n" + "comment: " + row['body'] + "\n" + "stance: "
    query = "[INST]" + query + "[/INST]"

    output = text_gen(f"<s>{query}]")
    generated_text = output[0]['generated_text']
    predicted_label = generated_text.split('[/INST]')[1].split('</s>')[0]
    data.at[index, 'stance'] = predicted_label
    
    # Save every 1000 rows
    if ii % 100 == 0:
        print(generated_text)
        # Create a backup of the old save
        if os.path.exists(save_path):
            shutil.copyfile(save_path, backup_path)
        
        # Save the data to the path
        data.to_pickle(save_path)
    ii += 1

if os.path.exists(save_path):
    shutil.copyfile(save_path, backup_path)
            
# Save the data to the path
data.to_pickle(save_path)