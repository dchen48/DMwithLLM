# DMwithLLM
## OneShot-wikilinks
### How to get the data

* download from https://www.kaggle.com/generall/oneshotwikilinks
* unzip
* run `cut -f1 shuffled_dedup_entities.tsv | sort -S50% | uniq -c | sort -S10% -k1rn > entityfreq.gz`
* python make_data.py

### Run experiments
* Pure bandit: python bandit.py 
* Corral (clipping): python corral_clip --llm_type [small,base,large] --min_prob=[0, 0.1, 0.2]
* Corral (mixing): python corral_mix --llm_type large --gamma [0.05,0.1,0.2,0.4]
* Corral (early stopping): python corral_early_stopping --llm_type large --max_num_llms [10000,20000]
* Corral with equation 1: python corral_eq1 --llm_type large --max_num_llms [10000,20000]
* Corral with multiple LLMs: python corral_multiple_llms.py --min_prob=0.2 
* Linear decay: Python linear_decay --llm_type large --init_prob 0.8 
* Exponential decay: python exp_decay --llm_type large --init_prob 0.8 --beta [0.1,0.01] --c_exp [1,10,100]
* Polynomial decay:  python poly_decay --llm_type large --init_prob 0.8 --beta [0.1,0.01] --c_poly [1,10,100]
* LLM as decision making agent: python llm_agent --llm_type [small,base,large]
* bandit learned with purely LLM selected data: python bandit_all_llm.py --llm_type large

## AmazonCat-13K
### How to get the data

* download AmazonCat-13K dataset from the publicly available Extreme Classification Repository:    http://manikvarma.org/downloads/XC/XMLRepository.html
* unzip
* python make_data.py

### Run experiments
* Pure bandit: python bandit.py 
* Corral (clipping): python corral_clip --llm_type small --min_prob=0.2
* LLM as decision making agent: python llm_agent --llm_type small

## Acknowledgement
* The code for SpannerGreedy is modified based on https://github.com/pmineiro/linrepcb/blob/main/oneshotwikilinks/spannerepsgreedy.ipynb.
* The code for Corral is modified based on https://github.com/pmineiro/smoothcb/blob/main/contextualbanditexperiment/tune-fastcbcorral.py

