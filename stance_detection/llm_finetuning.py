import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import (LoraConfig, AutoPeftModelForCausalLM, PeftModel)
from trl import SFTTrainer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

print(f'running on cuda device {torch.cuda.current_device()}')
max_memory={0: "0.0GB", 1: "0GB", 2: "0GB", 3: "22GB", 4: "22GB", 5: "0GB", 6: "0GB", 7: "0GB", 'cpu': "10GB"}

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

base_text_gen = pipeline(task="text-generation", model=base_model, tokenizer=llama_tokenizer, max_new_tokens=5)

query_head = "You are a helpful, respectful, and honest assistant that detects the stance of a comment with respect to its parent. Stance detection is the process of determining whether the author of a comment is in support of or against a given parent. You are provided with:\n post: the text you that is the root of discussion.\n parent:  the text which the comment is a reply towards.\n comment: text that you identify the stance from.\n\nYou will return the stance of the comment against the parent. Only return the stance against the parent and not the original post. Always answer from the possible options given below: \n support: The comment has a positive or supportive attitude towards the post, either explicitly or implicitly. \n against: The comment opposes or criticizes the post, either explicitly or implicitly. \n none: The comment is neutral or does not have a stance towards the post. \n unsure: It is not possible to make a decision based on the information at hand."
#query = f"<SYS> {query_head} </SYS>" + "\n\n" + "post: " + row['submission_text'] + "\n" + "parent: " + row['body_parent'] + "\n" + "comment: " + row['body_child'] + "\n" + "stance: "
#query = "[INST]" + query + "[/INST]"


training_data = load_from_disk('stance_detection/curated_datasets/debagreement_data_10000/train')
print(training_data)

# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=16,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training Params
train_params = TrainingArguments(
    output_dir="./results_modified",
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=100,
    learning_rate=1e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.04,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# Trainer
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    args=train_params
)

## %tensorboard --logdir logs/fit

# Training
fine_tuning.train()
refined_model = "stance-detection/llama-2-7b-reddit_stance_det_lora" #You can give it your own name
fine_tuning.model.save_pretrained(refined_model)