"""類似点と相違点の文章化を行う."""

from tqdm import tqdm
from typing import List
import numpy as np
import pandas as pd
from langchain.chat_models import ChatOpenAI
from utils import messages, update_progress


def difference_verbalization(config):
    dataset = config['output_dir']
    path = f"outputs/{dataset}/difference_verbalization.csv"

    arguments = pd.read_csv(f"outputs/{dataset}/args.csv")
    clusters = pd.read_csv(f"outputs/{dataset}/clusters.csv")

    results = pd.DataFrame()

    sample_size = config['difference_verbalization']['sample_size']

    prompt = config['difference_verbalization']['prompt']
    #similarity_prompt = config['verbalization']['similarity_prompt']
    #difference_prompt = config['verbalization']['difference_prompt']

    model = config['difference_verbalization']['model']

    cluster_ids = clusters['cluster-id'].unique()

    update_progress(config, total=len(cluster_ids))

    for _, cluster_id in tqdm(enumerate(cluster_ids), total=len(cluster_ids)):
        args_ids = clusters[clusters['cluster-id']
                            == cluster_id]['arg-id'].values
        args_ids = np.random.choice(args_ids, size=min(
            len(args_ids), sample_size), replace=False)
        # コーパスA
        args_sample = arguments[(arguments['arg-id'].isin(args_ids)) & (arguments['categoryLabel'] == 1)]['argument'].values
        # コーパスB
        args_sample_outside = arguments[(arguments['arg-id'].isin(args_ids)) & (arguments['categoryLabel'] == 0)]['argument'].values
        
        difference = generate_difference_verbalization(args_sample,
                               args_sample_outside, prompt, model)        
        results = pd.concat([results, pd.DataFrame(
            [{'cluster-id': cluster_id, 'difference':difference}])], ignore_index=True)
        update_progress(config, incr=1)

    results.to_csv(path, index=False)


def generate_difference_verbalization(args_sample, args_sample_outside, prompt, model):
    llm = ChatOpenAI(model_name=model, temperature=0.0)
    copus_a = '\n * ' + '\n * '.join(args_sample_outside)
    copus_b = '\n * ' + '\n * '.join(args_sample)
    input = f"Examples of arguments copus_a the cluster:\n {copus_a}" + \
        f"Examples of arguments copus_b the cluster:\n {copus_b}"
    response = llm(messages=messages(prompt, input)).content.strip()
    return response