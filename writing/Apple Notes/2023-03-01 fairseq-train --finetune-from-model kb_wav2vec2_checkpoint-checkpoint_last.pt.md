fairseq-train --finetune-from-model kb_wav2vec2_checkpoint/[checkpoint_last.pt](http://checkpoint_last.pt)

--train-subset sbtal_subset.tsv
--valid-subset fairseq-valid.tsv
--tensorboard-logdir fairseq-tensorboard



common.tensorboard_logdir