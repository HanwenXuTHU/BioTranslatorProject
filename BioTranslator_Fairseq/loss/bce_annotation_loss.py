import torch
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('bce_annotation_loss')
class BCEAnnotationLoss(FairseqCriterion):
    def __init__(self, task):
        super().__init__(task)
        self.label_k = task.label_k
        self.features = task.features
        self.loss_func = torch.nn.BCELoss()

    def forward(self, model, sample, reduce=True):
        input_list = [sample[f] for f in self.features]
        logits = model(input_list, self.text_emb)
        loss = self.loss_func(logits, sample[self.label_k])
        sample_size = (sample[self.label_k].size(0))
        logging_output = {
            "loss": loss.data,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def set_text_emb(self, text_emb):
        self.text_emb = text_emb

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log["loss"].item() for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum, sample_size, round=3)