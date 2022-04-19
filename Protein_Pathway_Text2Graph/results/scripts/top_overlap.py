import collections
import numpy as np
import pandas as pd


def main():
    top_n = 50
    overlap_path = "/home/hwxu/BioTranslator/Protein_Pathway_Prediction/results/description_sort/Reactome_goa_human_cat_for_pathway.csv"
    overlap = pd.read_csv(overlap_path)
    top_overlap = collections.OrderedDict()
    overlap_words = list(overlap['overlap'])
    sents = list(overlap['pathway_description'])
    overlap_count = [len(overlap_words[i].split(', ')) for i in range(len(overlap_words))]
    overlap_count = np.asarray(overlap_count)
    top_idx = np.argsort(-overlap_count)[0 : top_n]
    top_overlap = overlap.loc[top_idx]
    if 'Unnamed: 0' in top_overlap.columns:
        top_overlap.drop(['Unnamed: 0'], axis=1, inplace=True)
    top_overlap.to_csv('../description_sort/selected_description_overlap.csv', index=False)


if __name__ == '__main__':
    main()