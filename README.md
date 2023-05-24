# BioTranslator

## Section 1: Introduction
BioTranslator is a cross-modal translator which can annotate biology instances only using user-written texts.
The codes here can reproduce the main results in BioTranslator paper, including zero-shot protein function prediction, zero-shot cell type prediction, and predict the nodes and edges of a gene pathway.
BioTranslator takes a user-written textual description of the new discovery and then translates this description to a non-text biological data instance. Our tool frees scientists from limiting their analysis within predefined controlled vocabularies, thus accelerating new biomedical discoveries.

**Update 08/05/2022:** We add the [fairseq](https://fairseq.readthedocs.io) version of BioTranslator. Please refer to the [BioTranslator_Fairseq/scripts/](https://github.com/HanwenXuTHU/BioTranslatorProject/tree/main/BioTranslator_Fairseq/scripts) for how to perform the protein function prediction and cell type discovery. 

**Update 04/11/2023:** We now released our [BioTranslator](https://biotranslator.readthedocs.io/en/latest/index.html) python package and tutorials!

## Section 2: Installation Tutorial
### Section 2.1: System Requirement
BioTranslator is implemented using Python 3.7 in LINUX. BioTranslator requires torch==1.7.1+cu110, torchvision==0.8.2+cu110, numpy, pandas, sklearn, transformers, networkx, seaborn, tokenizers and so on.
BioTranslator requires you have one GPU device to run the codes.
You can install the packages by:
```cmd
python3.7 -m pip install --upgrade pip
pip install -r requirements.txt
``` 
### Section 2.2: How to use our codes
The function annotation, cell type discovery and pathway analysis task in our paper are put in Protein, SingleCell and Pathway respectively.
The main codes are in the BioTranslator folder and the struture of the project is
- Structure of BioTranslator

``` 
BioTranslator/  
├── __init__.py/
├── BioConfig.py/ 
├── BioLoader.py/ 
├── BioMetrics.py/ 
├── BioModel.py/ 
├── BioTrain.py/   
├── BioUtils.py/  
```
The first step of BioTranslator is to train a text encoder with contrastive learning on 225 ontologies data.

### Section 2.3 Train a text encoder
First please download the [Graphine](https://zenodo.org/record/5320310#.YUBtu55Kgox) dataset and unzip it. 
Then you need to specify the path where you unzip the Graphine dataset and the path you save the trained text encoder in TextEncoder/train_text_encoder.py.

For example, 
```python
# the path where you save the model
save_model = 'model/text_encoder.pth'
# the path where you store the data
graphine_repo = '/data/Graphine/dataset/'
```
The training process will take several hours, please wait patiently or you can directly download the trained text encoder [model](https://figshare.com/articles/dataset/Protein_Pathway_data_tar/20120447).
### Section 2.4 New Functions Annotation
You can run Protein/main.py to reproduce results of protein function prediction. We provide the command line interface.
We also release the codes of baselines here. You can specify which method you like to run. First please download the Protein_Pathway dataset provided in our paper
and unzip it. The following command lines can reproduce our results in the zero shot task. For example, you can run BioTranslator on the GOA (Human) dataset.
```cmd
python Protein/main.py --method BioTranslator --dataset GOA_Human --data_repo /data/ProteinDataset --task zero_shot --encoder_path model/text_encoder.pth --emb_path /embeddings
```
+ **'method'**: Specify the method to run, choose between BioTranslator, ProTranslator, TFIDF, clusDCA, Word2Vec, Doc2Vec.
+ **'dataset'**: Specify the dataset for cross-validation, choose between GOA_Human, GOA_Mouse, GOA_Yeast, SwissProt, CAFA3.
+ **'data_repo'**: Where you store the potein dataset, this folder should contains GOA_Human, GOA_Mouse, GOA_Yeast, SwissProt, CAFA3 folder.
+ **'task'**: Choose between zero_shot task and few_shot task.
+ **'encoder_path'**: The path of text encoder model.
+ **'emb_path'**: Where you cache the textual description embeddings.
The results will be save in the working_space/task/results/ folder with the following structure.
- The structure of results folder

``` 
working_space/  
├── zero_shot/  
│   ├── log/  
│   ├── results/ 
|   └── model/
└── few_shot/  
    ├── log/  
    ├── results/ 
    └── model/ 
```
The inference results will be saved in results/$method$_$dataset$.pkl.
Then you can run the codes on different dataset.
```cmd
python Protein/main.py --method BioTranslator --dataset GOA_Human --data_repo /data/ProteinDataset --task zero_shot --encoder_path model/text_encoder.pth --emb_path /embeddings
python Protein/main.py --method BioTranslator --dataset GOA_Mouse --data_repo /data/ProteinDataset --task zero_shot --encoder_path model/text_encoder.pth --emb_path /embeddings
python Protein/main.py --method BioTranslator --dataset GOA_Yeast --data_repo /data/ProteinDataset --task zero_shot --encoder_path model/text_encoder.pth --emb_path /embeddings
python Protein/main.py --method BioTranslator --dataset SwissProt --data_repo /data/ProteinDataset --task zero_shot --encoder_path model/text_encoder.pth --emb_path /embeddings
python Protein/main.py --method BioTranslator --dataset CAFA3 --data_repo /data/ProteinDataset --task zero_shot --encoder_path model/text_encoder.pth --emb_path /embeddings
```
Run the codes using different baselines/
```cmd
python Protein/main.py --method BioTranslator --dataset GOA_Human --data_repo /data/ProteinDataset --task zero_shot --encoder_path model/text_encoder.pth --emb_path /embeddings
python Protein/main.py --method ProTranslator --dataset GOA_Human --data_repo /data/ProteinDataset --task zero_shot --encoder_path model/text_encoder.pth --emb_path /embeddings
python Protein/main.py --method TFIDF --dataset GOA_Human --data_repo /data/ProteinDataset --task zero_shot --encoder_path model/text_encoder.pth --emb_path /embeddings
python Protein/main.py --method clusDCA --dataset GOA_Human --data_repo /data/ProteinDataset --task zero_shot --encoder_path model/text_encoder.pth --emb_path /embeddings
python Protein/main.py --method Word2Vec --dataset GOA_Human --data_repo /data/ProteinDataset --task zero_shot --encoder_path model/text_encoder.pth --emb_path /embeddings
python Protein/main.py --method Doc2Vec --dataset GOA_Human --data_repo /data/ProteinDataset --task zero_shot --encoder_path model/text_encoder.pth --emb_path /embeddings
```
Run the codes to perform the few shot prediction task. 
```cmd
python Protein/main.py --method BioTranslator --dataset GOA_Human --data_repo /data/ProteinDataset --task few_shot --encoder_path model/text_encoder.pth --emb_path /embeddings
```
The results of few shot task with blast will be saved in results/$method$_$dataset$_blast.pkl.
### Section 2.5 New Cell Type Discovery
You can run SingleCell/main.py to reproduce results of new cell type annotation. 
```cmd
python SingleCell/main.py --dataset muris_droplet --data_repo /data/sc_data --task same_dataset --encoder_path model/text_encoder.pth --emb_path /embeddings
```
+ **'dataset'**: Specify the dataset for cross-validation, choose between sapiens, tabula_microcebus, muris_droplet, microcebusAntoine, microcebusBernard, microcebusMartine, microcebusStumpy, muris_facs.
+ **'data_repo'**: Where you store the single cell dataset.
+ **'task'**: Choose between same_dataset task and cross_dataset task. same_dataset: cross-validation on the same dataset. cross_dataset: cross-dataset validation.
+ **'encoder_path'**: The path of text encoder model.
+ **'emb_path'**: Where you cache the textual description embeddings.
The results will be save in the working_space/task/results/ folder.
- The structure of results folder

``` 
working_space/  
├── one_dataset/  
│   ├── log/  
│   ├── results/ 
|   └── model/
└── cross_dataset/  
    ├── log/  
    ├── results/ 
    └── model/ 
```
You can also run the codes to reproduce the results of cross-dataset validation.
```cmd
python SingleCell/main.py --dataset muris_droplet --data_repo /data/sc_data --task cross_dataset --encoder_path model/text_encoder.pth --emb_path /embeddings
```
### Section 2.6 Pathway Analysis
In this section, we will show how to predict the nodes and links in a pathway.
You can run Pathway/main.py to perform pathway analysis.
```cmd
python Pathway/main.py --pathway_dataset KEGG --dataset GOA_Human --data_repo /data/Protein_Pathway_data/ --encoder_path model/text_encoder.pth --emb_path /embeddings
python Pathway/main.py --pathway_dataset KEGG --dataset GOA_Human --data_repo /data/Protein_Pathway_data/ --encoder_path model/text_encoder.pth --emb_path /embeddings
python Pathway/main.py --pathway_dataset KEGG --dataset GOA_Human --data_repo /data/Protein_Pathway_data/ --encoder_path model/text_encoder.pth --emb_path /embeddings
```
+ **'pathway_dataset'**: The pathway dataset, choose between Reactome, KEGG and PharmGKB
+ **'dataset'**: The dataset you choose to train our BioTranslator. In our paper, we set dataset to GOA_Human
+ **'data_repo'**: Where you store the potein dataset and pathway dataset, this folder should contains Reactome, KEGG, PharmGKB, GOA_Human, GOA_Mouse, GOA_Yeast, SwissProt, CAFA3 folder.
+ **'encoder_path'**: The path of text encoder model.
+ **'emb_path'**: Where you cache the textual description embeddings.
This code contains to step: (1) train BioTranslator. (2) perform node classification and edge prediction.
The results are
```cmd
train terms number:21656
eval pathway number:337
Rank of your embeddings is 768
Rank of your embeddings is 337
Data Loading Finished!
Start training model on :GOA_Human ...
initialize network with xavier
Training: 100%|███████████████████████| 30/30 [25:58<00:00, 51.95s/it, epoch=29, train loss=0.00187]
Evaluate Our Model on KEGG
Pathway Node Classification: 100%|████████████████████████████████| 209/209 [00:24<00:00,  8.70it/s]
2022-06-22 02:23:30,612 - BioTrainer.py[line:124] - INFO: Pathway: KEGG Node Classification AUROC: 0.7438614121231653
Pathway Edge Prediction: 100%|████████████████████████████████████| 337/337 [04:13<00:00,  1.33it/s]
2022-06-22 02:27:43,656 - BioTrainer.py[line:170] - INFO: Pathway: KEGG Edge Prediction AUROC: 0.7894009929801727
```

The authors are trying to make BioTranslator easy-to-use, but it's impossible to include every detail of our algorithm in one document.
So if you have any question about the software, feel free to contact us (xuhanwenthu@gmail.com).
