from transformers import T5Tokenizer, T5TokenizerFast, PreTrainedTokenizer, PreTrainedTokenizerBase
import re
import numpy as np
import sentencepiece as spm

# The special tokens of T5Tokenizer is hard-coded with <extra_id_{}>
# Created another class UDOPTokenizer extending it to add special visual tokens like <loc_{}>, etc.

class UdopTokenizer(T5Tokenizer):

    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        loc_extra_ids=501,
        other_extra_ids=200,
        additional_special_tokens=[],
        sp_model_kwargs=None,
        **kwargs
    ):
        # Add extra_ids to the special token list
        if extra_ids > 0 and not "<extra_id_0>" in additional_special_tokens:
            additional_special_tokens = ["<extra_id_{}>".format(i) for i in range(extra_ids)]
            additional_special_tokens.extend(["<extra_l_id_{}>".format(i) for i in range(extra_ids)])
            additional_special_tokens.extend(["</extra_l_id_{}>".format(i) for i in range(extra_ids)])
            additional_special_tokens.extend(["<extra_t_id_{}>".format(i) for i in range(extra_ids)])
            additional_special_tokens.extend(["</extra_t_id_{}>".format(i) for i in range(extra_ids)])

        if loc_extra_ids > 0 and not "<loc_0>" in additional_special_tokens:
            additional_special_tokens.extend(["<loc_{}>".format(i) for i in range(loc_extra_ids)])

        if other_extra_ids > 0 and not "<other_0>" in additional_special_tokens:
            additional_special_tokens.extend(["<other_{}>".format(i) for i in range(other_extra_ids)])
        print(additional_special_tokens)
        PreTrainedTokenizer.__init__(
            self,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
        self._loc_extra_ids = loc_extra_ids
        self._other_extra_ids = other_extra_ids

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)
        
    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size() + self._extra_ids * 5 + self._loc_extra_ids + self._other_extra_ids

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(
            i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 4
        elif token.startswith("<extra_l_id_"):
            match = re.match(r"<extra_l_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 3
        elif token.startswith("</extra_l_id_"):
            match = re.match(r"</extra_l_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 2
        elif token.startswith("<extra_t_id_"):
            match = re.match(r"<extra_t_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids - self._loc_extra_ids - self._extra_ids
        elif token.startswith("</extra_t_id_"):
            match = re.match(r"</extra_t_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids - self._loc_extra_ids
        elif token.startswith("<loc_"):
            match = re.match(r"<loc_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids
        elif token.startswith("<other_"):
            match = re.match(r"<other_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1
        
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index < self.sp_model.get_piece_size():
            token = self.sp_model.IdToPiece(index)
        else:
            
            if index > self.sp_model.get_piece_size() + self._extra_ids * 5 + self._loc_extra_ids - 1:
                index_loc = self.vocab_size - 1 - index
                token = f"<other_{index_loc}>"           
            elif index > self.sp_model.get_piece_size() + self._extra_ids * 5 - 1:
                index_loc = self.vocab_size - self._other_extra_ids - 1 - index
                token = f"<loc_{index_loc}>"   
            elif index > self.sp_model.get_piece_size() + self._extra_ids * 4 - 1:
                token = "</extra_t_id_{}>".format(self.vocab_size - self._other_extra_ids - self._loc_extra_ids - 1 - index)
            elif index > self.sp_model.get_piece_size() + self._extra_ids * 3 - 1:
                token = "<extra_t_id_{}>".format(self.vocab_size - self._other_extra_ids - self._loc_extra_ids - self._extra_ids - 1 - index)
            elif index > self.sp_model.get_piece_size() + self._extra_ids * 2 - 1:
                token = "</extra_l_id_{}>".format(self.vocab_size - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 2 - 1 - index)
            elif index > self.sp_model.get_piece_size() + self._extra_ids - 1:
                token = "<extra_l_id_{}>".format(self.vocab_size - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 3 - 1 - index)
            elif index > self.sp_model.get_piece_size() - 1:
                token = "<extra_id_{}>".format(self.vocab_size - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 4 - 1 - index)
            else:
                raise
        return token
    
    def get_bbox_from_logits(self, logits):
        """
        Exctact bbox of sentence given logits of shape (batch_size, seq_length) 
        to bbox of shape (batch_size, 4).
        Logits must be normalized with softmax and argmax.
        If the seq doesn't contain bbox, return [-1, -1, -1, -1]
        """
        bbox = np.ones((logits.shape[0], 4), dtype=np.int32)
        # Get rows of logits that contain bbox
        loc_inx_start = self.vocab_size - self._loc_extra_ids - self._other_extra_ids
        loc_inx_end = self.vocab_size - self._other_extra_ids
        loc_inx = np.where(np.logical_and(logits >= loc_inx_start, logits < loc_inx_end))

        # Get unique rows
        rows_bbox = np.unique(loc_inx[0])

        for i in range(logits.shape[0]):
            if i in rows_bbox:
                # Get index where loc_inx[0] == i
                inx = np.where(loc_inx[0] == i)
                # Get logits of bbox
                logits_bbox = logits[i, loc_inx[1][inx]]
                # Get bbox
                bbox[i, :] = logits_bbox - loc_inx_end + 1

        bbox = bbox * -1
        return bbox
        

    def convert_bbox_to_token(self, bbox, page_size):
        # Convert to tokens depending of localization tokens vocab size. 
        # Example: normalized bbox: [0.1, 0.2, 0.5, 0.6], vocab size: 500 -> [<loc_50><loc_100><loc_250><loc_300>]
        bbox = [bbox[0] / page_size[0], bbox[1] / page_size[1], bbox[2] / page_size[0], bbox[3] / page_size[1]]
        
        tokens = []
        for b in bbox:
            tokens.append(f'<loc_{int(b * (self._loc_extra_ids - 1))}>')
        return tokens

    def convert_token_to_bbox(self, tokens, page_size):
        # Convert tokens to bbox depending of localization tokens vocab size. 
        # Example: [<loc_50><loc_100><loc_250><loc_300>], vocab size: 500 -> [0.1, 0.2, 0.5, 0.6]
        bbox = []
        for t in tokens:
            match = re.match(r"<loc_(\d+)>", t)
            num = int(match.group(1))
            bbox.append(num / (self._loc_extra_ids - 1))
        return [bbox[0] * page_size[0], bbox[1] * page_size[1], bbox[2] * page_size[0], bbox[3] * page_size[1]]


# Below are for Rust-based Fast Tokenizer

from transformers.convert_slow_tokenizer import SpmConverter
from tokenizers import processors
from typing import List


class UdopConverter(SpmConverter):
    def vocab(self, proto):
        vocab = [(piece.piece, piece.score) for piece in proto.pieces]
        num_extra_ids = self.original_tokenizer._extra_ids
        vocab += [("<extra_id_{}>".format(i), 0.0)
                  for i in range(num_extra_ids - 1, -1, -1)]
        vocab += [("<extra_l_id_{}>".format(i), 0.0)
                  for i in range(num_extra_ids - 1, -1, -1)]
        vocab += [("</extra_l_id_{}>".format(i), 0.0)
                  for i in range(num_extra_ids - 1, -1, -1)]
        vocab += [("<extra_t_id_{}>".format(i), 0.0)
                  for i in range(num_extra_ids - 1, -1, -1)]
        vocab += [("</extra_t_id_{}>".format(i), 0.0)
                  for i in range(num_extra_ids - 1, -1, -1)]
        
        num_loc_extra_ids = self.original_tokenizer._loc_extra_ids
        vocab += [("<loc_{}>".format(i), 0.0)
                  for i in range(num_loc_extra_ids - 1, -1, -1)]

        num_other_extra_ids = self.original_tokenizer._other_extra_ids
        vocab += [("<other_0{}>".format(i), 0.0)
                  for i in range(num_other_extra_ids - 1, -1, -1)]
        
        return vocab

    def post_processor(self):
        return processors.TemplateProcessing(
            single=["$A", "</s>"],
            pair=["$A", "</s>", "$B", "</s>"],
            special_tokens=[
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


def convert_slow_udoptokenizer(UdopTokenizer):
    return UdopConverter(UdopTokenizer).converted()


class UdopTokenizerFast(T5TokenizerFast):

    slow_tokenizer_class = UdopTokenizer
    prefix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file,
        tokenizer_file=None,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        loc_extra_ids=201,
        other_extra_ids=200,
        additional_special_tokens=None,
        **kwargs
    ):
        # Add extra_ids to the special token list
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = ["<extra_id_{}>".format(i) for i in range(extra_ids)]
            additional_special_tokens.extend(["<extra_l_id_{}>".format(i) for i in range(extra_ids)])
            additional_special_tokens.extend(["</extra_l_id_{}>".format(i) for i in range(extra_ids)])
            additional_special_tokens.extend(["<extra_t_id_{}>".format(i) for i in range(extra_ids)])
            additional_special_tokens.extend(["</extra_t_id_{}>".format(i) for i in range(extra_ids)])

        if loc_extra_ids > 0 and not "<loc_0>" in additional_special_tokens:
            additional_special_tokens.extend(["<loc_{}>".format(i) for i in range(loc_extra_ids)])
            
        if other_extra_ids > 0 and not "<other_0>" in additional_special_tokens:
            additional_special_tokens.extend(["<other_{}>".format(i) for i in range(other_extra_ids)])
        
        slow_tokenizer = self.slow_tokenizer_class(
            vocab_file,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            loc_extra_ids=loc_extra_ids,
            other_extra_ids=other_extra_ids,
            **kwargs
        )
        fast_tokenizer = convert_slow_udoptokenizer(slow_tokenizer)
        self._tokenizer = fast_tokenizer

        PreTrainedTokenizerBase.__init__(
            self,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
        self._loc_extra_ids = loc_extra_ids
        self._other_extra_ids = other_extra_ids
