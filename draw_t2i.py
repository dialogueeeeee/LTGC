from clip_filter import clip_filter
from dalle_gen import dalle_gen, get_cls_index_name, description_refine, get_cls_template
import pandas as pd
from openai import OpenAI
import clip
import torch
import os
import argparse


# hyper-param
parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('-ext', '--extended_description_path', default='descriptions_data/extended_description.csv', type=str,
                    help='File path to the extended description file')
parser.add_argument('-d', '--data_dir', default=200, type=int,
                    help='Directory for the dataset')
parser.add_argument('-t', '--thresh', default=0.6, type=int,
                    help='Threshold for clip filter')
parser.add_argument('-r', '--max_rounds', default=3, type=int,
                    help='Max round for clip filter')
args = parser.parse_args()


client = OpenAI(api_key='Replace with your own OPENAI KEY.')


df = pd.read_csv(args.extended_description_path, header=None, names=['label', 'text'])
grouped_list = df.groupby('label')['text'].apply(list).to_dict()

for label, texts in grouped_list.items():
    index_name = get_cls_index_name(label)
    dir_path = os.path.join(args.data_dir, 'gen_train', str(index_name))
    os.makedirs(dir_path, exist_ok=True)
    
    for text_i in range(len(texts)):
        saved_path = os.path.join(dir_path, f"{index_name}_{text_i}.JPEG")

        img_path = dalle_gen(client, saved_path, texts[text_i], saved=True)
        if img_path != None:

            ## clip filter
            score = 0

            for round in range(args.max_rounds):
                if round > 0:
                    refine_texts = description_refine(texts[text_i], index_name)
                    saved_path_refine = os.path.join(dir_path, f"{index_name}_{text_i}_refine{round}.JPEG")
                    img_path = dalle_gen(client, saved_path_refine, refine_texts, saved=True)
                    if img_path != None:
                        score = clip_filter(img_path, cls_feature_template)
                else:
                    cls_feature_template = get_cls_template(index_name, label)
                    score = clip_filter(img_path, cls_feature_template)

                if score >= args.thresh-0.5:
                    break

