# Classifying heat pump-related innovations in patents and research abstracts

The scripts in this folder are to be run in this order:

**(1) `filter_relevance.py`**

This script collects a balanced sample of research abstracts and patents and uses GPT to classify them as relevant or not relevant to heat pumps. The output is a CSV file with the classification results.

With modification, it can include an optional keyword filtering step prior to the GPT classification step.

**(2) `classify_innovations.py`**

This script takes just those abstracts and patents that were classed as relevant in step (1), and extracts more detailed information about them.
