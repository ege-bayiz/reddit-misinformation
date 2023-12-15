import pandas as pd
from sklearn.metrics import confusion_matrix

stance_dict = {'against': 0, 'none': 1, 'support': 2, 'unsure' : -1}
stance_dict_inv = {0: 'against', 1: 'none', 2: 'support'}

df = pd.read_csv('stance_detection/curated_datasets/debagreement_gpt4_with_stance.csv')
df_gt = pd.read_csv('stance_detection/DEBAGREEMENT_data/Labeled Dataset/labeled_data.csv')

df['stance'] = df['stance'].apply(lambda x: stance_dict[x])
print(df_gt)

print(confusion_matrix(df['stance'], df_gt['label']))