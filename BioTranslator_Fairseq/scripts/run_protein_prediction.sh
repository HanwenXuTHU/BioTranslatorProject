TASK=protein_zero_shot
ARCH=biotranslator_arch
CRITERION=bce_annotation_loss

DATANAME=GOA_Human

ENCODERPATH=/data/xuhw/research-project/BioTranslatorCode/TextEncoder/Encoder/encoder.pth
EMBPATH=/data/xuhw/data_emb/
DATADIR=/data/xuhw/data/ProteinDataset
SAVEDIR=/data/xuhw/research-project/BioTranslator_Fairseq/$TASK/ckpt

mkdir -p $SAVEDIR

python /data/xuhw/research-project/BioTranslator_Fairseq/train.py $DATADIR \
  --task $TASK \
  --arch $ARCH \
  --criterion $CRITERION \
  --encoder-path $ENCODERPATH \
  --emb-path $EMBPATH \
  --data-name $DATANAME \
  --optimizer adam \
  --lr 3e-4 --lr-scheduler fixed \
  --batch-size 32 \
  --max-epoch 30 \
  --distributed-world-size 1 \
  --max-tokens 500 \
  --gpu-ids 1 \
  --save-dir $SAVEDIR | tee -a $SAVEDIR/train.log \
