# -*- coding:utf-8 -*-
import os
import functools
from typing import Any, Dict, List, Union, Iterable
import torch
from mylogger import logger
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel, AutoConfig
from hanlp.datasets.tokenization.loaders.txt import TextTokenizingDataset, generate_tags_for_subtokens
from hanlp.utils.span_util import bmes_to_spans
from hanlp.layers.embeddings.embedding import EmbeddingDim, Embedding
from hanlp.transform.transformer_tokenizer import TransformerSequenceTokenizer
from hanlp.utils.torch_util import lengths_to_mask
from hanlp.layers.crf.crf import CRF
from hanlp_trie import TrieDict
from hanlp.layers.transformers.encoder import TransformerEncoder
from hanlp_common.structure import SerializableDict
from hanlp.common.transform import FieldLength, TransformList, VocabDict, EmbeddingNamedTransform

IDX = '_idx_'


def dtype_of(e: Union[int, bool, float]):
    if isinstance(e, bool):
        return torch.bool
    if isinstance(e, int):
        return torch.long
    if isinstance(e, float):
        return torch.float
    raise ValueError(f'Unsupported type of {repr(e)}')


def merge_dict(d: dict, overwrite=False, inplace=False, **kwargs):
    """Merging the provided dict with other kvs

    Args:
      d:
      kwargs:
      d: dict:
      overwrite:  (Default value = False)
      inplace:  (Default value = False)
      **kwargs:

    Returns:


    """
    nd = dict([(k, v) for k, v in d.items()] + [(k, v) for k, v in kwargs.items() if overwrite or k not in d])
    if inplace:
        d.update(nd)
        return d
    return nd


def merge_list_of_dict(samples: List[Dict]) -> dict:
    batch = {}
    for each in samples:
        for k, v in each.items():
            vs = batch.get(k, None)
            if vs is None:
                vs = []
                batch[k] = vs
            vs.append(v)
    return batch


def reorder(samples: List, order: List[int]) -> List:
    return [samples[i] for i in sorted(range(len(order)), key=lambda k: order[k])]


class PadSequenceDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=32, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None,
                 pad: dict = None, vocabs: VocabDict = None, device=None, **kwargs):
        """ A dataloader commonly used for NLP tasks. It offers the following convenience.

        - Bachify each field of samples into a :class:`~torch.Tensor` if the field name satisfies the following criterion.
            - Name ends with _id, _ids, _count, _offset, _span, mask
            - Name is in `pad` dict.

        - Pad each field according to field name, the vocabs and pad dict.
        - Move :class:`~torch.Tensor` onto device.

        Args:
            dataset: A :class:`~torch.utils.data.Dataset` to be bachified.
            batch_size: Max size of each batch.
            shuffle: ``True`` to shuffle batches.
            sampler: A :class:`~torch.utils.data.Sampler` to sample samples from data.
            batch_sampler: A :class:`~torch.utils.data.Sampler` to sample batches form all batches.
            num_workers: Number of workers for multi-thread loading. Note that multi-thread loading aren't always
                faster.
            collate_fn: A function to perform batchifying. It must be set to ``None`` in order to make use of the
                 features this class offers.
            pin_memory: If samples are loaded in the Dataset on CPU and would like to be pushed to
                    the GPU, enabling pin_memory can speed up the transfer. It's not useful since most data field are
                    not in Tensor type.
            drop_last: Drop the last batch since it could be half-empty.
            timeout: For multi-worker loading, set a timeout to wait for a worker.
            worker_init_fn: Init function for multi-worker.
            multiprocessing_context: Context for multiprocessing.
            pad: A dict holding field names and their padding values.
            vocabs: A dict of vocabs so padding value can be fetched from it.
            device: The device tensors will be moved onto.
            **kwargs: Other arguments will be passed to :meth:`torch.utils.data.Dataset.__init__`
        """
        if device == -1:
            device = None
        if collate_fn is None:
            collate_fn = self.collate_fn

        if batch_sampler is None:
            assert batch_size, 'batch_size has to be specified when batch_sampler is None'
        else:
            batch_size = 1
            shuffle = None
            drop_last = None
        # noinspection PyArgumentList
        super(PadSequenceDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                                    sampler=sampler,
                                                    batch_sampler=batch_sampler, num_workers=num_workers,
                                                    collate_fn=collate_fn,
                                                    pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
                                                    worker_init_fn=worker_init_fn,
                                                    multiprocessing_context=multiprocessing_context, **kwargs)
        self.vocabs = vocabs
        if dataset.transform:
            transform = dataset.transform
            if not isinstance(transform, TransformList):
                transform = []
            for each in transform:
                if isinstance(each, EmbeddingNamedTransform):
                    if pad is None:
                        pad = {}
                    if each.dst not in pad:
                        pad[each.dst] = 0
        self.pad = pad
        self.device = device

    def __iter__(self):
        for raw_batch in super(PadSequenceDataLoader, self).__iter__():
            yield self.tensorize(raw_batch, vocabs=self.vocabs, pad_dict=self.pad, device=self.device)

    @staticmethod
    def tensorize(raw_batch: Dict[str, Any], vocabs: VocabDict, pad_dict: Dict[str, int] = None, device=None):
        for field, data in raw_batch.items():
            if isinstance(data, torch.Tensor):
                continue
            vocab_key = field[:-len('_id')] if field.endswith('_id') else None
            vocab = vocabs.get(vocab_key, None) if vocabs and vocab_key else None
            if vocab:
                pad = vocab.safe_pad_token_idx
                dtype = torch.long
            elif pad_dict is not None and pad_dict.get(field, None) is not None:
                pad = pad_dict[field]
                dtype = dtype_of(pad)
            elif field.endswith('_offset') or field.endswith('_id') or field.endswith(
                    '_count') or field.endswith('_ids') or field.endswith('_score') or field.endswith(
                '_length') or field.endswith('_span'):
                # guess some common fields to pad
                pad = 0
                dtype = torch.long
            elif field.endswith('_mask'):
                pad = False
                dtype = torch.bool
            else:
                # no need to pad
                continue
            data = PadSequenceDataLoader.pad_data(data, pad, dtype)
            raw_batch[field] = data
        if device is not None:
            for field, data in raw_batch.items():
                if isinstance(data, torch.Tensor):
                    data = data.to(device)
                    raw_batch[field] = data
        return raw_batch

    @staticmethod
    def pad_data(data: Union[torch.Tensor, Iterable], pad, dtype=None, device=None):
        """Perform the actual padding for a given data.

        Args:
            data: Data to be padded.
            pad: Padding value.
            dtype: Data type.
            device: Device to be moved onto.

        Returns:
            torch.Tensor: A ``torch.Tensor``.
        """
        if isinstance(data[0], torch.Tensor):
            data = pad_sequence(data, True, pad)
        elif isinstance(data[0], Iterable):
            inner_is_iterable = False
            for each in data:
                if len(each):
                    if isinstance(each[0], Iterable):
                        inner_is_iterable = True
                        if len(each[0]):
                            if not dtype:
                                dtype = dtype_of(each[0][0])
                    else:
                        inner_is_iterable = False
                        if not dtype:
                            dtype = dtype_of(each[0])
                    break
            if inner_is_iterable:
                max_seq_len = len(max(data, key=len))
                max_word_len = len(max([chars for words in data for chars in words], key=len))
                ids = torch.zeros(len(data), max_seq_len, max_word_len, dtype=dtype, device=device)
                for i, words in enumerate(data):
                    for j, chars in enumerate(words):
                        ids[i][j][:len(chars)] = torch.tensor(chars, dtype=dtype, device=device)
                data = ids
            else:
                data = pad_sequence([torch.tensor(x, dtype=dtype, device=device) for x in data], True, pad)
        elif isinstance(data, list):
            data = torch.tensor(data, dtype=dtype, device=device)
        return data

    def collate_fn(self, samples):
        return merge_list_of_dict(samples)


class TransformerTaggingModel(nn.Module):
    def __init__(self,
                 encoder: TransformerEncoder,
                 num_labels,
                 crf=False,
                 secondary_encoder=None,
                 extra_embeddings: EmbeddingDim = None) -> None:
        """
        A shallow tagging model use transformer as decoder.
        Args:
            encoder: A pretrained transformer.
            num_labels: Size of tagset.
            crf: True to enable CRF.
            extra_embeddings: Extra embeddings which will be concatenated to the encoder outputs.
        """
        super().__init__()
        self.encoder = encoder
        self.secondary_encoder = secondary_encoder
        self.extra_embeddings = extra_embeddings
        # noinspection PyUnresolvedReferences
        feature_size = encoder.transformer.config.hidden_size
        if extra_embeddings:
            feature_size += extra_embeddings.get_output_dim()
        self.classifier = nn.Linear(feature_size, num_labels)
        self.crf = CRF(num_labels) if crf else None

    def forward(self, lens: torch.LongTensor, input_ids, token_span, token_type_ids=None, batch=None):
        mask = lengths_to_mask(lens)
        x = self.encoder(input_ids, token_span=token_span, token_type_ids=token_type_ids)
        if self.secondary_encoder:
            x = self.secondary_encoder(x, mask=mask)
        if self.extra_embeddings:
            # noinspection PyCallingNonCallable
            embed = self.extra_embeddings(batch, mask=mask)
            x = torch.cat([x, embed], dim=-1)
        x = self.classifier(x)
        return x, mask


class ChineseWordSegmentation:

    def __init__(self, model_name_or_path, device='cuda', **kwargs):
        self.dict_force = TrieDict()
        self.dict_combine = TrieDict()
        self.stop_words_path = os.path.join(model_name_or_path, 'stop_words.txt')
        self.stop_words = self._load_stop_words()
        self._tokenizer_transform = None
        self.config = SerializableDict(**kwargs)
        self.vocabs = VocabDict()
        self.device = torch.device("cuda" if (torch.cuda.is_available() and device.startswith("cuda")) else "cpu")
        self.transformer_tokenizer = None
        self.model = None
        self.load(model_name_or_path)

    def _load_stop_words(self):
        stop_words = []
        if os.path.exists(self.stop_words_path):
            with open(self.stop_words_path, 'r', encoding='utf-8') as reader:
                for line in reader:
                    line = line.strip()
                    stop_words.append(line)
        return stop_words

    def _save_stop_words(self):
        try:
            with open(self.stop_words_path, 'w', encoding='utf-8') as f:
                for i in self.stop_words:
                    f.write(i + '\n')
        except Exception as e:
            logger.error({'EXCEPTION': e})

    def add_stopwords(self, stopwords_list: List):
        self.stop_words.extend(stopwords_list)
        self.stop_words = list(set(self.stop_words))
        self._save_stop_words()

    def add_keywords_force(self, keywords_list, clear=False):
        # Examples:
        #     >>> tok.dict_force = {'和服', '服务行业'} # Force '和服' and '服务行业' by longest-prefix-matching
        #     >>> tok("商品和服务行业")
        #         ['商品', '和服', '务行业']
        #     >>> tok.dict_force = {'和服务': ['和', '服务']} # Force '和服务' to be tokenized as ['和', '服务']
        #     >>> tok("商品和服务行业")
        #         ['商品', '和', '服务', '行业']
        if clear:
            self.dict_force = TrieDict()

        for k in keywords_list:
            if isinstance(k, dict):
                self.dict_force.update(k)
            else:
                self.dict_force.update({k: True})

        self.tokenizer_transform.dict = self.dict_force

    def add_keywords_combine(self, keywords_list, clear=False):
        # Examples:
        #     >>> tok.dict_combine = {'和服', '服务行业'}
        #     >>> tok("商品和服务行业") # '和服' is not in the original results ['商品', '和', '服务']. '服务', '行业' are combined to '服务行业'
        #         ['商品', '和', '服务行业']

        if clear:
            self.dict_combine = TrieDict()

        self.dict_combine.update({k: True for k in set(keywords_list)})

    def decode_output(self, logits, mask, batch):
        if self.config.get('crf', False):
            output = self.model.crf.decode(logits, mask)
        else:
            output = logits.argmax(-1)

        if isinstance(output, torch.Tensor):
            output = output.tolist()
        prediction = self.id_to_tags(output, [len(x) for x in batch['token']])
        return self.tag_to_span(prediction, batch)

    def tag_to_span(self, batch_tags, batch: dict):
        spans = []
        if 'custom_words' in batch:
            if self.config.tagging_scheme == 'BMES':
                S = 'S'
                M = 'M'
                E = 'E'
            else:
                S = 'B'
                M = 'I'
                E = 'I'
            for tags, custom_words in zip(batch_tags, batch['custom_words']):
                # [batch['raw_token'][0][x[0]:x[1]] for x in subwords]
                if custom_words:
                    for start, end, label in custom_words:
                        if end - start == 1:
                            tags[start] = S
                        else:
                            tags[start] = 'B'
                            tags[end - 1] = E
                            for i in range(start + 1, end - 1):
                                tags[i] = M
                        if end < len(tags):
                            tags[end] = 'B'
        if 'token_subtoken_offsets_group' not in batch:  # only check prediction on raw text for now
            # Check cases that a single char gets split into multiple subtokens, e.g., ‥ -> . + .
            for tags, subtoken_offsets in zip(batch_tags, batch['token_subtoken_offsets']):
                offset = -1  # BERT produces 'ᄒ', '##ᅡ', '##ᆫ' for '한' and they share the same span
                prev_tag = None
                for i, (tag, (b, e)) in enumerate(zip(tags, subtoken_offsets)):
                    if b < offset:
                        if prev_tag == 'S':
                            tags[i - 1] = 'B'
                        elif prev_tag == 'E':
                            tags[i - 1] = 'M'
                        tags[i] = 'M'
                    offset = e
                    prev_tag = tag
        for tags in batch_tags:
            spans.append(bmes_to_spans(tags))
        return spans

    @property
    def tokenizer_transform(self):
        if not self._tokenizer_transform:
            self._tokenizer_transform = TransformerSequenceTokenizer(self.transformer_tokenizer,
                                                                     self.config.token_key,
                                                                     ret_subtokens=True,
                                                                     ret_subtokens_group=True,
                                                                     ret_token_span=False,
                                                                     dict_force=self.dict_force)
        return self._tokenizer_transform

    def spans_to_tokens(self, spans, batch, rebuild_span=False):
        batch_tokens = []
        dict_combine = self.dict_combine
        raw_text = batch.get('token_', None)  # Use raw text to rebuild the token according to its offset
        for b, (spans_per_sent, sub_tokens) in enumerate(zip(spans, batch[self.config.token_key])):
            if raw_text:  # This will restore iPhone X as a whole
                text = raw_text[b]
                offsets = batch['token_subtoken_offsets'][b]
                tokens = [text[offsets[b][0]:offsets[e - 1][-1]] for b, e in spans_per_sent]
            else:  # This will merge iPhone X into iPhoneX
                tokens = [''.join(sub_tokens[span[0]:span[1]]) for span in spans_per_sent]

            if dict_combine:
                buffer = []
                offset = 0
                delta = 0
                for start, end, label in dict_combine.tokenize(tokens):
                    if offset < start:
                        buffer.extend(tokens[offset:start])
                    if raw_text:
                        # noinspection PyUnboundLocalVariable
                        combined = text[offsets[spans_per_sent[start - delta][0]][0]:
                                        offsets[spans_per_sent[end - delta - 1][1] - 1][1]]
                    else:
                        combined = ''.join(tokens[start:end])
                    buffer.append(combined)
                    offset = end
                    if rebuild_span:
                        start -= delta
                        end -= delta
                        combined_span = (spans_per_sent[start][0], spans_per_sent[end - 1][1])
                        del spans_per_sent[start:end]
                        delta += end - start - 1
                        spans_per_sent.insert(start, combined_span)
                if offset < len(tokens):
                    buffer.extend(tokens[offset:])
                tokens = buffer
            batch_tokens.append(tokens)
        return batch_tokens

    def prediction_to_human(self, pred, batch, rebuild_span=False):
        output_spans = self.config.get('output_spans', None)
        tokens = self.spans_to_tokens(pred, batch, rebuild_span or output_spans)
        if output_spans:
            subtoken_spans = batch['token_subtoken_offsets']
            results = []
            for toks, offs, subs in zip(tokens, pred, subtoken_spans):
                r = []
                results.append(r)
                for t, (b, e) in zip(toks, offs):
                    r.append([t, subs[b][0], subs[e - 1][-1]])
            return results
        return tokens

    def build_dataset(self, data, **kwargs):
        return TextTokenizingDataset(data, **kwargs)

    def last_transform(self):
        return TransformList(functools.partial(generate_tags_for_subtokens, tagging_scheme=self.config.tagging_scheme),
                             TransformList(self.vocabs, FieldLength(self.config.token_key)))

    def feed_batch(self, batch: dict):
        features = [batch[k] for k in self.tokenizer_transform.output_key]
        if len(features) == 2:
            input_ids, token_span = features
        else:
            input_ids, token_span = features[0], None
        lens = batch[f'{self.config.token_key}_length']
        x, mask = self.model(lens, input_ids, token_span, batch.get(f'{self.config.token_key}_token_type_ids'),
                             batch=batch)
        return x[:, 1:-1, :], mask

    def load_weights(self, save_dir, filename='model.pt', **kwargs):
        filename = os.path.join(save_dir, filename)
        self.model.load_state_dict(torch.load(filename, map_location='cpu'), strict=False)

    def load_config(self, save_dir, filename='config.json', **kwargs):
        self.config.load_json(os.path.join(save_dir, filename))
        self.config.update(kwargs)  # overwrite config loaded from disk
        self.config.transformer = AutoModel.from_config(
            AutoConfig.from_pretrained(os.path.join(save_dir, 'model')))
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(os.path.join(save_dir, 'model'), use_fast=True)

    def load_vocabs(self, save_dir, filename='vocabs.json'):
        if hasattr(self, 'vocabs'):
            self.vocabs = VocabDict()
            self.vocabs.load_vocabs(save_dir, filename)

    def load(self, save_dir: str, **kwargs):
        self.load_config(save_dir, **kwargs)
        self.load_vocabs(save_dir)
        self.model = self.build_model(
            **merge_dict(self.config, **kwargs, overwrite=True, inplace=True), training=False, save_dir=save_dir)

        self.load_weights(save_dir, **kwargs)
        self.model.to(self.device)
        self.model.eval()

    def build_transformer(self, training=True):
        transformer = TransformerEncoder(self.config.transformer, self.transformer_tokenizer,
                                         self.config.average_subwords,
                                         self.config.scalar_mix, self.config.word_dropout,
                                         ret_raw_hidden_states=self.config.ret_raw_hidden_states,
                                         training=training)
        transformer_layers = self.config.get('transformer_layers', None)
        if transformer_layers:
            transformer.transformer.encoder.layer = transformer.transformer.encoder.layer[:transformer_layers]
        return transformer

    def id_to_tags(self, ids: torch.LongTensor, lens: List[int]):
        batch = []
        vocab = self.vocabs['tag'].idx_to_token
        for b, l in zip(ids, lens):
            batch.append([])
            for i in b[:l]:
                batch[-1].append(vocab[i])
        return batch

    @torch.no_grad()
    def predict(self, sentences: List[str], return_origin=False, batch_size: int = 32, **kwargs):

        if isinstance(sentences, str):
            sentences = [sentences]

        samples = [{self.config.token_key: sent} for sent in sentences]
        if not batch_size:
            batch_size = self.config.get('batch_size', 32)
        dataloader = self.build_dataloader(data=samples, batch_size=batch_size, **kwargs)
        outputs = []
        orders = []
        for batch in dataloader:
            out, mask = self.feed_batch(batch)
            pred = self.decode_output(out, mask, batch)
            outputs.extend(self.prediction_to_human(pred, batch))
            orders.extend(batch[IDX])

        outputs = reorder(outputs, orders)
        if return_origin:
            return outputs

        outputs = [[w for w in o if w not in self.stop_words and len(w) > 1] for o in outputs]

        return outputs

    def build_model(self, training=False, extra_embeddings: Embedding = None, **kwargs) -> torch.nn.Module:
        model = TransformerTaggingModel(
            self.build_transformer(training=training),
            len(self.vocabs.tag),
            self.config.crf,
            self.config.get('secondary_encoder', None),
            extra_embeddings=extra_embeddings.module(self.vocabs) if extra_embeddings else None,
        )
        return model

    def build_dataloader(self, data, batch_size=32, **kwargs) -> DataLoader:
        args = dict((k, self.config.get(k, None)) for k in
                    ['delimiter', 'max_seq_len', 'sent_delimiter', 'char_level', 'hard_constraint'])
        dataset = self.build_dataset(data, **args)

        dataset.append_transform(self.tokenizer_transform)
        dataset.append_transform(self.last_transform())

        return PadSequenceDataLoader(dataset=dataset, batch_size=batch_size, device=self.device)
