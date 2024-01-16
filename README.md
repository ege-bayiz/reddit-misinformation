# The Effect of Uncredible News on User Clusters on Reddit
Analysis of uncredible and highly biased news in Reddit user clusters. This is a condensed version of the repository with some of the required data embedded in it for review purposes.

## How to run the codes?

### Stance detection training and validation
The folder `stance_detection/` includes codes for stance detection training and validation together with the condensed copy of the DEBAGREEMENT dataset we explain in the paper (see `curated_datasets/` folder). Run `llm_finetuning.py` to fine tune a LoRA model on the LLaMa2 model. For validation of the results run `llm_validation.py` which generates a pandas dataframe consisting of ground trith labels and preficted labels. To pass reddit comment-post pairs through the stance detection model run `comment_stances.py`


### Negation fitting
The folder `user_embedding/` includes codes for both negation training and clustering analysis. For training the Affine Negation model run `sentence_negation_fitting.py`. The `sentence_negation_preprocessing.py` if for the processing of the negation dataset. We include an already preprocessed copy of the negation dataset for convenience of review.


### Cluster analysis
Lastly, all of the user clustering and analysis are in a single python notebook `user_embedding/user_embedding.ipynb`. This data requires the input of reddit user comment and post embedding dataframes. The posts need to be pre-embedded with the `all-distilroberta-v1` model which can be found here (https://huggingface.co/sentence-transformers/all-distilroberta-v1). They also need to be preassigned with news credibility and biases from the Ad Fontes Media dataset (https://adfontesmedia.com/).

