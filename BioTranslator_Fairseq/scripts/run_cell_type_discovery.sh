TASK=cell_type_discovery
ARCH=biotranslator_arch
CRITERION=bce_annotation_loss
OntologyDir=/data/xuhw/data/Ontology_data/

DATANAME=muris_droplet

ENCODERPATH=/data/xuhw/research-project/BioTranslatorCode/TextEncoder/Encoder/encoder.pth
EMBPATH=/data/xuhw/data_emb/
BACKUPFILE=../backup_file/backup.h5ad

DATADIR=/data/xuhw/data/sc_data/
SAVEDIR=/data/xuhw/research-project/BioTranslator_Fairseq/$TASK/ckpt

mkdir -p $SAVEDIR

python /data/xuhw/research-project/BioTranslator_Fairseq/train.py $DATADIR \
  --task $TASK \
  --arch $ARCH \
  --criterion $CRITERION \
  --encoder-path $ENCODERPATH \
  --ontology-repo $OntologyDir \
  --backup-file $BACKUPFILE \
  --emb-path $EMBPATH \
  --data-name $DATANAME \
  --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
  --lr 1e-4 --lr-scheduler fixed \
  --unseen-ratio 0.5 \
  --batch-size 128 \
  --max-epoch 15 \
  --distributed-world-size 1 \
  --max-tokens 500 \
  --gpu-ids 1 \
  --save-dir $SAVEDIR | tee -a $SAVEDIR/train.log \
