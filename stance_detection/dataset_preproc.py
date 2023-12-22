import pandas as pd
from datasets import Dataset
from tqdm import tqdm

stance_dict = {'against': 0, 'none': 1, 'favor': 2}
stance_dict_inv = {0: 'against', 1: 'none', 2: 'favor'}

df = pd.read_csv('stance_detection/DEBAGREEMENT_data/Labeled Dataset/labeled_data.csv')

samples = df.loc[df['agreement_fraction'] > 0.5]
samples = samples.sample(n=10000, random_state=0)
print(samples['label'].value_counts())

prompt_texts = []
for index, row in tqdm(samples.iterrows(), total=len(samples), desc="Processing samples"):
    query_head = "You are a helpful, respectful, and honest assistant that detects the stance of a comment with respect to its parent. Stance detection is the process of determining whether the author of a comment is in support of or against a given parent. You are provided with:\n post: the text you that is the root of discussion.\n parent:  the text which the comment is a reply towards.\n comment: text that you identify the stance from.\n\nYou will return the stance of the comment against the parent. Only return the stance against the parent and not the original post. Always answer from the possible options given below: \n support: The comment has a positive or supportive attitude towards the post, either explicitly or implicitly. \n against: The comment opposes or criticizes the post, either explicitly or implicitly. \n none: The comment is neutral or does not have a stance towards the post. \n unsure: It is not possible to make a decision based on the information at hand."
    query = f"<SYS> {query_head} </SYS>" + "\n\n" + "post: " + row['submission_text'] + "\n" + "parent: " + row['body_parent'] + "\n" + "comment: " + row['body_child'] + "\n" + "stance: "
    query = "[INST]" + query + "[/INST]"
    prompt_texts.append("<s>" + query + stance_dict_inv[row['label']] + "</s>")

data = Dataset.from_dict({'text': prompt_texts})
data = data.train_test_split(test_size=0.1)
print(data)

data.save_to_disk("stance_detection/curated_datasets/debagreement_data_10000")
#print(dataset_entry)

unlabeled_df = df[['submission_text', 'body_parent', 'body_child']]

unlabeled_df.to_csv('stance_detection/curated_datasets/debagreement_gpt4.csv', index=False)