## Evaluate accuracy of the relevance filtering step

- Run `pipeline/filter_relevance.py` on a sample

- Run `pipeline/evaluation/create_relevance_sample.py` to create a balanced sample of 10 predicted positive, 10 predicted negative examples each for OpenAlex and for patents

- Manually label this in Google Sheets or similar

- Download the labelled data store it in `inputs/`

- Calculate accuracy

## Evaluate the classification step

- Run `pipeline/filter_relevance.py` and `pipeline/classify_innovations.py` on the full data. Some of the categories are quite sparse, so we need the full data in order to have a good chance of getting enough positive examples for each category/subcategory.

- Run `pipeline/evaluation/create_balanced_sample.py`: this creates a balanced sample from the data labelled by GPT at the previous step. This script aims to get 5 positive and 5 negative examples per subcategory.

- `format_sample_for_labelling.py`: format the sample so that it is as straightforward as possible to manually annotate.

- Manually label this in Google Sheets or similar. For most categories, go one column at a time and label 5x predicted 0s and 5x predicted 1s. For application type (and ground source/air source if applicable) try to label 5x "industrial", 5x "domestic, 5x "both" and 5x "unclear". Address TRL last as this will likely take the most brain power!

- Download the labelled data store it in `inputs/`

- Generate evaluation metrics (initially precision, and later we can add recall and F1)
