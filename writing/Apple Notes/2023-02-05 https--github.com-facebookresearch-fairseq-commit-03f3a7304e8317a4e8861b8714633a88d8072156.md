https://github.com/facebookresearch/fairseq/commit/03f3a7304e8317a4e8861b8714633a88d8072156

class DummyMultiTask(LegacyFairseqTask):
    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        if self.args.decoder_type == "ctc":
            model = models\[0\]  # only support single model
            encoder_out = model(**sample)
            if hasattr(model, "get_logits"):
                emissions = model.get_logits(
                    encoder_out
                )  # no need to normalize emissions
            else:
                emissions = model.get_normalized_probs(encoder_out, log_probs=True)
            return generator.decode(
                emissions.transpose(0, 1).float().cpu().contiguous()
            )
        else:
            raise NotImplementedError("only ctc decoder is supported at the moment")

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        if self.args.decoder_type == "ctc":
            from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder

            return W2lViterbiDecoder(args, self.tgt_dict)
        else:
            raise NotImplementedError("only ctc decoder is supported at the moment")