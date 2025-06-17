## Evaluate accuracy of the relevance filtering step

- Run `pipeline/filter_relevance.py` on a sample

- Run `pipeline/evaluation/create_relevance_sample.py` to create a balanced sample of 10 predicted positive, 10 predicted negative examples each for OpenAlex and for patents

- Manually label this in Google Sheets or similar

- Download the labelled data store it in `inputs/`

- Calculate accuracy

## Evaluate the classification step

- Run `pipeline/filter_relevance.py` and `pipeline/classify_innovations.py`

- `pipeline/evaluation/create_balanced_sample.py`: sample data from those instances already labelled by GPT in order to create a balanced sample

- `format_sample_for_labelling.py`: format the sample so that it is as straightforward as possible to manually annotate

- Manually label this in Google Sheets or similar

- Download the labelled data store it in `inputs/`

- Generate evaluation metrics (initially precision, and later we can add recall and F1)
