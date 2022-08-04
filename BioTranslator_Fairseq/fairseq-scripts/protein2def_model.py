import torch
from torch import nn
from fairseq.models.transformer import TransformerDecoder
from fairseq.models import FairseqModel, register_model, BaseFairseqModel
from fairseq import utils
from typing import Any, Dict, List, Optional, NamedTuple
from torch import Tensor


EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Optional[Tensor]),  # B x T
        ("encoder_embedding", Optional[Tensor]),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("src_tokens", Optional[Tensor]),  # B x T
        ("src_lengths", Optional[Tensor]),  # B x 1
    ],
)

@register_model('protein2def_model')
class Protein2DefModel(BaseFairseqModel):

    def __init__(self, args, task, decoder):
        super(Protein2DefModel, self).__init__()
        self.decoder = decoder
        max_kernels = 33
        self.para_conv, self.para_pooling = [], []
        self.src_embedding = self.build_embedding(args, task.source_dictionary, embed_dim=args.encoder_embed_dim)
        self.padding_idx = self.src_embedding.padding_idx
        self.in_nc = args.encoder_embed_dim
        kernels = range(8, max_kernels, 4)
        self.input_nc = args.encoder_embed_dim
        self.kernel_num = len(kernels)
        seqL = args.max_positions[0]
        for i in range(len(kernels)):
            exec("self.conv1d_{} = nn.Conv1d(in_channels=args.encoder_embed_dim, out_channels=self.in_nc, kernel_size=kernels[i], padding=0, stride=1)".format(i))
            exec("self.pool1d_{} = nn.MaxPool1d(kernel_size=seqL - kernels[i] + 1, stride=1)".format(i))

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--decoder-embed-dim', type=int, metavar='N',
            help='dimensionality of the decoder embeddings',
        )
        parser.add_argument(
            '--decoder-hidden-dim', type=int, metavar='N',
            help='dimensionality of the decoder hidden state',
        )
        parser.add_argument(
            '--decoder-dropout', type=float, default=0.1,
            help='decoder dropout probability',
        )

    @classmethod
    def build_model(cls, args, task):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a SimpleLSTMModel instance.

        # Initialize our Encoder and Decoder.

        decoder_tokens = cls.build_embedding(cls, args=args, dictionary=task.target_dictionary, embed_dim=args.decoder_embed_dim)
        decoder = TransformerDecoder(args=args, dictionary=task.target_dictionary, embed_tokens=decoder_tokens)
        model = Protein2DefModel(args, task, decoder)

        # Print the model architecture.
        print(model)

        return model

    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = torch.nn.Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = False,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        x = src_tokens
        encoder_padding_mask = torch.zeros([src_tokens.size(0), self.kernel_num]).cuda()
        x = self.src_embedding(x)
        x = x.permute(0, 2, 1)
        x_list = []
        for i in range(self.kernel_num):
            exec("x_i = self.conv1d_{}(x)".format(i))
            exec("x_i = torch.mean(x_i, dim=2)".format(i))
            exec("x_list.append(torch.squeeze(x_i))")
        x_cat = torch.cat(tuple(x_list), dim=1)
        encoder_out = x_cat.reshape([-1, self.kernel_num, self.in_nc])
        encoder_out = EncoderOut(encoder_out=[encoder_out.transpose(0, 1)],
                                 encoder_embedding=[encoder_out],
                                 encoder_padding_mask=[encoder_padding_mask],
                                 encoder_states=None,
                                 src_tokens=None,
                                 src_lengths=None
                                 )._asdict()
        new_src_lengths = self.kernel_num*torch.ones([src_tokens.size(0)])
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=new_src_lengths,
            return_all_hiddens=return_all_hiddens,
        )

        return decoder_out

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to `new_order`.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            `encoder_out` rearranged according to `new_order`
        """
        final_hidden = encoder_out['final_hidden']
        return {
            'final_hidden': final_hidden.index_select(0, new_order),
        }


from fairseq.models import register_model_architecture

# The first argument to ``register_model_architecture()`` should be the name
# of the model we registered above (i.e., 'simple_lstm'). The function we
# register here should take a single argument *args* and modify it in-place
# to match the desired architecture.


@register_model_architecture('protein2def_model', 'protein2def_arch')
def protein2def_arch(args):
    # We use ``getattr()`` to prioritize arguments that are explicitly given
    # on the command-line, so that the defaults defined below are only used
    # when no other value has been specified.
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 256)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_hidden_dim = getattr(args, 'decoder_hidden_dim', 256)



