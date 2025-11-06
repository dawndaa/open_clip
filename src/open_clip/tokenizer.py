""" CLIP tokenizer

Copied from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import gzip
import html
import os
import random
import string
from functools import lru_cache, partial
from typing import Callable, List, Optional, Union, Dict
import warnings

import ftfy
import numpy as np
import regex as re
import torch

# https://stackoverflow.com/q/62691279
os.environ["TOKENIZERS_PARALLELISM"] = "false"
_nltk_init = False

DEFAULT_CONTEXT_LENGTH = 77  # default context length for OpenAI CLIP


@lru_cache()
def default_bpe() -> str:
    """返回默认的BPE词表文件路径。

    English: Return the default BPE vocabulary path shipped with OpenCLIP.

    Returns:
        str: 默认BPE词表文件的绝对路径 (absolute path)。
    """

    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode() -> Dict[int, str]:
    """构造UTF-8字节到Unicode字符的可逆映射表。

    English: Build reversible lookup tables between UTF-8 bytes and Unicode strings
    so that the BPE merges can operate purely on textual tokens without hitting
    control characters.

    Returns:
        Dict[int, str]: 字节数值到Unicode字符的映射 (byte-to-unicode map)。
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """枚举单词中相邻符号对集合。

    English: Return the set of adjacent symbol pairs within a token sequence.

    Args:
        word (Tuple[str, ...]): 当前单词的符号序列 (token tuple)。

    Returns:
        Set[Tuple[str, str]]: 所有可能的相邻符号对 (adjacent pairs)。
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text: str) -> str:
    """执行基础文本清洗以矫正编码并去除多余空白。

    English: Fix Unicode issues, unescape HTML entities, and trim whitespace.

    Args:
        text (str): 原始输入文本 (raw text).

    Returns:
        str: 清洗后的字符串 (cleaned text).
    """

    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    """压缩空白字符并移除首尾空格。

    English: Collapse repeated whitespace and strip leading/trailing spaces.

    Args:
        text (str): 输入字符串 (input text).

    Returns:
        str: 处理后的字符串 (whitespace-normalized text).
    """

    text = " ".join(text.split())
    text = text.strip()
    return text


def _clean_canonicalize(x: str) -> str:
    """执行规范化清洗（去标点、统一大小写）。

    English: Apply canonicalization with punctuation removal and lowercasing.
    """

    # 基础处理 + 去除空白和标点，统一为小写
    return canonicalize_text(basic_clean(x))


def _clean_lower(x: str) -> str:
    """执行清洗后转小写的流程。

    English: Clean text and convert to lowercase while keeping punctuation.
    """

    # 基础处理 + 去除空白，并保持标点
    return whitespace_clean(basic_clean(x)).lower()


def _clean_whitespace(x: str) -> str:
    """仅做基础清洗并压缩空白。

    English: Clean text and collapse whitespace without changing case.
    """

    # 基础处理 + 去除空白
    return whitespace_clean(basic_clean(x))


def get_clean_fn(type: str) -> Callable[[str], str]:
    """根据清洗类型返回对应的文本处理函数。

    English: Dispatch the appropriate cleaning function based on configuration.

    Args:
        type (str): 清洗策略名称，例如 ``canonicalize``、``lower``。

    Returns:
        Callable[[str], str]: 具体的清洗函数 (clean function)。

    Raises:
        AssertionError: 当提供未知的清洗类型时触发。
    """

    if type == 'canonicalize':
        return _clean_canonicalize
    elif type == 'lower':
        return _clean_lower
    elif type == 'whitespace':
        return _clean_whitespace
    else:
        assert False, f"Invalid clean function ({type})."


def canonicalize_text(
    text: str,
    *,
    keep_punctuation_exact_string: Optional[str] = None,
    trans_punctuation: dict = str.maketrans("", "", string.punctuation),
) -> str:
    """规范化输入文本：转小写并移除标点。

    English: Canonicalize ``text`` by lowercasing and stripping punctuation while
    optionally preserving a specific substring. Logic adapted from Google's
    Big Vision project for prompt engineering.

    Args:
        text (str): 需要规范化的原始文本 (input text).
        keep_punctuation_exact_string (Optional[str]): 指定不被删除的精确子串。
            English: preserve this exact substring with punctuation.
        trans_punctuation (dict): ``str.translate`` 使用的标点映射。

    Returns:
        str: 规范化结果 (canonicalized text).
    """
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(trans_punctuation)
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(trans_punctuation)
    text = text.lower()
    text = " ".join(text.split())
    return text.strip()


class SimpleTokenizer(object):
    """面向OpenCLIP的BPE分词器实现。

    English: Implements the OpenCLIP-compatible byte pair encoding (BPE)
    tokenizer with optional text cleaning and reduction strategies.
    """

    def __init__(
            self,
            bpe_path: str = default_bpe(),
            additional_special_tokens: Optional[List[str]] = None,
            context_length: Optional[int] = DEFAULT_CONTEXT_LENGTH,
            clean: str = 'lower',
            reduction_mask: str = ''
    ):
        """初始化分词器并加载BPE词表数据。

        English: Initialize tokenizer internals and load merge rules.

        Args:
            bpe_path (str): BPE词表路径。
            additional_special_tokens (Optional[List[str]]): 额外特殊token列表。
            context_length (Optional[int]): 默认上下文长度 (context length)。
            clean (str): 文本清洗策略名称。
            reduction_mask (str): 可选的token裁剪策略标识。
        """

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        special_tokens = ['<start_of_text>', '<end_of_text>']
        if additional_special_tokens:
            special_tokens += additional_special_tokens
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t:t for t in special_tokens}
        special = "|".join(special_tokens)
        self.pat = re.compile(
            special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )
        self.vocab_size = len(self.encoder)
        self.all_special_ids = [self.encoder[t] for t in special_tokens]
        self.sot_token_id = self.all_special_ids[0]
        self.eot_token_id = self.all_special_ids[1]
        self.context_length = context_length
        self.clean_fn = get_clean_fn(clean)
        self.reduction_fn = get_reduction_mask_fn(reduction_mask) if reduction_mask else None

    def bpe(self, token: str) -> str:
        """对输入token执行BPE合并并返回子词序列。

        English: Apply cached BPE merge operations to produce a whitespace
        separated subword string.
        """

        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str) -> List[int]:
        """将文本编码为BPE token索引序列。

        English: Convert cleaned text into a list of token IDs.
        """

        bpe_tokens = []
        text = self.clean_fn(text)
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens: List[int]) -> str:
        """将token索引序列还原为字符串。

        English: Reconstruct original text from token IDs.
        """

        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

    def __call__(self, texts: Union[str, List[str]], context_length: Optional[int] = None) -> torch.LongTensor:
        """将字符串序列转换为固定长度的token张量。

        English: Tokenize one or multiple strings into a padded tensor of token
        IDs with ``sot``/``eot`` markers.

        Args:
            texts (Union[str, List[str]]): 待编码的文本或文本列表。
            context_length (Optional[int]): 上下文长度，默认取初始化时的值。

        Returns:
            torch.LongTensor: 形状 ``[batch, context_length]`` 的token张量。
        """
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length'

        if self.reduction_fn is not None:
            # use reduction strategy for tokenize if set, otherwise default to truncation below
            return self.reduction_fn(
                texts,
                context_length=context_length,
                sot_token_id=self.sot_token_id,
                eot_token_id=self.eot_token_id,
                encode_fn=self.encode,
            )

        all_tokens = [[self.sot_token_id] + self.encode(text) + [self.eot_token_id] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]  # Truncate
                tokens[-1] = self.eot_token_id
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result


_tokenizer = SimpleTokenizer()


def decode(output_ids: torch.Tensor) -> str:
    """使用默认分词器解码模型输出。

    English: Decode a tensor of token IDs via the global ``SimpleTokenizer``.

    Args:
        output_ids (torch.Tensor): 模型输出的token张量。

    Returns:
        str: 解码后的文本。
    """

    output_ids = output_ids.cpu().numpy()
    return _tokenizer.decode(output_ids)


def tokenize(texts: Union[str, List[str]], context_length: int = DEFAULT_CONTEXT_LENGTH) -> torch.LongTensor:
    """调用全局分词器对输入文本进行编码。

    English: Convenience wrapper around :class:`SimpleTokenizer`.
    """

    return _tokenizer(texts, context_length=context_length)


def random_mask_tokenize(
        texts: Union[str, List[str]],
        context_length: int,
        sot_token_id: int,
        eot_token_id: int,
        encode_fn: Callable,
        shuffle: bool = False,
):
    """基于随机掩码策略生成token张量。

    English: Tokenize with random token selection when exceeding context.

    Args:
        texts (Union[str, List[str]]): 输入文本集合。
        context_length (int): 目标上下文长度。
        sot_token_id (int): ``<start_of_text>`` 的索引。
        eot_token_id (int): ``<end_of_text>`` 的索引。
        encode_fn (Callable): 编码函数（通常为 ``SimpleTokenizer.encode``）。
        shuffle (bool): 是否在截断后打乱token顺序 (shuffle tokens)。

    Returns:
        torch.LongTensor: 形状 ``[batch, context_length]`` 的token张量。
    """

    all_tokens = [encode_fn(text) for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        tokens = torch.tensor(tokens)
        num_tokens = len(tokens)
        if num_tokens > context_length - 2:  # 2 for sot and eot token
            num_keep = context_length - 2
            indices = torch.randperm(len(tokens))
            indices = indices[:num_keep]
            if not shuffle:
                indices = indices.msort()
            tokens = tokens[indices]
            num_tokens = num_keep
        result[i, 0] = sot_token_id
        result[i, 1:num_tokens + 1] = tokens
        result[i, num_tokens + 1] = eot_token_id

    return result


def simple_mask_tokenize(
        texts: Union[str, List[str]],
        context_length: int,
        sot_token_id: int,
        eot_token_id: int,
        encode_fn: Callable,
):
    """使用滑动窗口方式裁剪超长文本。

    English: Tokenize by cropping a contiguous span when length exceeds budget.

    Args:
        texts (Union[str, List[str]]): 输入文本列表。
        context_length (int): 目标上下文长度。
        sot_token_id (int): ``sot`` token索引。
        eot_token_id (int): ``eot`` token索引。
        encode_fn (Callable): 文本编码函数。

    Returns:
        torch.LongTensor: Token张量。
    """

    all_tokens = [encode_fn(text) for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        num_tokens = len(tokens)
        if num_tokens > context_length - 2:  # 2 for sot and eot token
            num_keep = context_length - 2
            start_index = random.randint(0, num_tokens - num_keep)  # high is incl
            tokens = tokens[start_index: start_index + num_keep]
        tokens = [sot_token_id] + tokens + [eot_token_id]
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def syntax_mask_tokenize(
        texts: Union[str, List[str]],
        context_length: int,
        sot_token_id: int,
        eot_token_id: int,
        encode_fn: Callable,
) -> torch.LongTensor:
    """基于词性优先级的语法掩码策略进行分词。

    English: Apply part-of-speech guided selection before standard tokenization.

    Args:
        texts (Union[str, List[str]]): 输入文本集合。
        context_length (int): 目标上下文长度。
        sot_token_id (int): ``sot`` token索引。
        eot_token_id (int): ``eot`` token索引。
        encode_fn (Callable): 原始编码函数。

    Returns:
        torch.LongTensor: 形状 ``[batch, context_length]`` 的token张量。
    """
    import nltk
    global _nltk_init
    if not _nltk_init:
        # run them for the first time
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        _nltk_init = True

    def get_order(x):
        if x.startswith('NN'):
            return 1
        elif x.startswith('JJ'):
            return 2
        elif x.startswith('VB'):
            return 3
        else:
            return 4

    # syntax masking
    new_texts = []
    for text in texts:
        list_tokens = nltk.tokenize.word_tokenize(text)
        pos_tags = nltk.pos_tag(list_tokens)
        #  sample the words by get_order method
        order_list = [get_order(tag) for _, tag in pos_tags]
        sorted_ids = np.argsort(np.array(order_list))
        sampled_ids = sorted(sorted_ids[:context_length - 2]) # need 2 slots for sot and eot tokens
        sampled_tokens = np.take(np.array(list_tokens), sampled_ids, axis=0)  # sample the tokens

        new_text = ''
        for token in sampled_tokens:
            new_text = new_text + str(token) + ' '
        new_text = new_text.strip()
        new_texts.append(new_text)
    texts = new_texts

    all_tokens = [[sot_token_id] + encode_fn(text) + [eot_token_id] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        # still need first truncate because some words produces two tokens
        if len(tokens) > context_length:
            tokens = tokens[:context_length]  # Truncate
            tokens[-1] = eot_token_id
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def get_reduction_mask_fn(type: str) -> Callable:
    """根据策略选择token裁剪函数。

    English: Select a token reduction function to meet context constraints.

    Args:
        type (str): 策略名称，例如 ``simple``、``random``。

    Returns:
        Callable: 对应的裁剪函数。
    """

    assert type in ('simple', 'random', 'shuffle', 'syntax')
    if type == 'simple':
        return simple_mask_tokenize  # randomly select block [start:end]
    elif type == 'random':
        return random_mask_tokenize  # randomly drop tokens (keep order)
    elif type == 'shuffle':
        return partial(random_mask_tokenize, shuffle=True)  # randomly drop tokens (shuffle order)
    elif type == 'syntax':
        return syntax_mask_tokenize  # randomly drop prioritized by syntax
    else:
        assert False, f'Unknown type {type}.'


class HFTokenizer:
    """HuggingFace分词器封装，支持多种定制模式。

    English: Wrapper around HuggingFace tokenizers with OpenCLIP-specific
    cleaning and truncation utilities.
    """

    def __init__(
            self,
            tokenizer_name: str,
            context_length: Optional[int] = DEFAULT_CONTEXT_LENGTH,
            clean: str = 'whitespace',
            strip_sep_token: bool = False,
            language: Optional[str] = None,
            cache_dir: Optional[str] = None,
            tokenizer_mode: Optional[str] = None,  # None, 'clips'
            **kwargs
    ):
        """初始化HuggingFace分词器并应用OpenCLIP定制。

        English: Instantiate HuggingFace tokenizer with cleaning strategy,
        optional language selection, and CLIP-style modes.

        Args:
            tokenizer_name (str): HF模型名称或本地路径。
            context_length (Optional[int]): 默认上下文长度。
            clean (str): 文本清洗策略。
            strip_sep_token (bool): 是否移除 ``[SEP]`` token。
            language (Optional[str]): 指定源语言标签。
            cache_dir (Optional[str]): 模型缓存目录。
            tokenizer_mode (Optional[str]): 特殊模式，例如 ``clips``。
            **kwargs: 传递给 ``AutoTokenizer`` 的其他参数。
        """

        self.tokenizer_mode = tokenizer_mode or ''
        self.context_length = context_length
        self.clean_fn = get_clean_fn(clean)
        self.strip_sep_token = strip_sep_token

        # NOTE: Left as example of loading custom tokenizer from file for experimentation
        # if self.tokenizer_mode == 'bert_clips':
        #     self.special_tokens = {
        #         "bos_token": 1,
        #         "eos_token": 2,
        #         "cls_token": 101,
        #         "pad_token": 0
        #     }
        #
        #     # For BERT CLIPS mode with vocab file
        #     from tokenizers import BertWordPieceTokenizer
        #     if tokenizer_name.startswith('hf-hub:'):
        #         from huggingface_hub import hf_hub_download
        #         # Format: hf-hub:repo_id/filename
        #         repo_url = tokenizer_name[7:]
        #         parts = repo_url.split('/')
        #         filename = parts[-1]
        #         repo_id = '/'.join(parts[:-1])
        #         vocab_file = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        #         self.tokenizer = BertWordPieceTokenizer(lowercase=True)
        #         self.tokenizer = self.tokenizer.from_file(vocab_file)
        #     else:
        #         # Assume tokenizer_name is a local path to a vocab file
        #         self.tokenizer = BertWordPieceTokenizer(lowercase=True)
        #         self.tokenizer = self.tokenizer.from_file(tokenizer_name)

        # Standard HuggingFace tokenizer initialization
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
            **kwargs
        )

        # Set language function if available
        set_lang_fn = getattr(self.tokenizer, 'set_src_lang_special_tokens', None)
        if callable(set_lang_fn):
            self.set_lang_fn = set_lang_fn
        if language is not None:
            self.set_language(language)

    def save_pretrained(self, dest: str) -> None:
        """将内部tokenizer保存到目标目录。

        English: Persist underlying tokenizer assets for reuse.
        """

        self.tokenizer.save_pretrained(dest)

    def __call__(self, texts: Union[str, List[str]], context_length: Optional[int] = None) -> torch.Tensor:
        """执行批量分词并对齐到统一长度。

        English: Tokenize text batch with optional CLIP-style behavior.
        """
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length in class init or call.'

        texts = [self.clean_fn(text) for text in texts]

        # Handle different tokenization modes
        if self.tokenizer_mode == 'clips':
            return self._clips_tokenize(texts, context_length)
        else:
            # Standard tokenization
            input_ids = self.tokenizer.batch_encode_plus(
                texts,
                return_tensors='pt',
                max_length=context_length,
                padding='max_length',
                truncation=True,
            ).input_ids

            if self.strip_sep_token:
                input_ids = torch.where(
                    input_ids == self.tokenizer.sep_token_id,
                    torch.zeros_like(input_ids),
                    input_ids,
                )

            return input_ids

    def set_language(self, src_lang: str) -> None:
        """配置源语言以启用多语言特殊token。

        English: Forward language selection to HuggingFace tokenizer when
        supported.
        """

        if hasattr(self, 'set_lang_fn'):
            self.set_lang_fn(src_lang)
        else:
            warnings.warn('Cannot set language for the tokenizer.')

    def _clips_tokenize(self, texts: List[str], context_length: int) -> torch.Tensor:
        """使用标准HF分词器并施加CLIP风格后处理。

        English: Emulate original CLIP token layout with HuggingFace tokenizers.
        """
        # Use standard tokenizer without special tokens - we'll add our own
        encoded_outputs = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors=None
        )

        encoded = []
        for tokens in encoded_outputs["input_ids"]:
            tokens = tokens[:context_length - 3]  # Leave room for special tokens
            tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
            encoded.append(tokens)

        # Create result tensor and handle padding + class token
        result = torch.zeros(len(encoded), context_length, dtype=torch.long)
        for i, tokens in enumerate(encoded):
            padded_tokens = self._pad_and_add_class_token(
                tokens,
                max_length=context_length,
                pad_token_id=self.tokenizer.pad_token_id,
                cls_token_id=self.tokenizer.cls_token_id,
            )
            result[i, :len(padded_tokens)] = torch.tensor(padded_tokens)

        return result

    def _pad_and_add_class_token(
            self,
            tokens: List[int],
            max_length: int,
            pad_token_id: int = 0,
            cls_token_id: int = 101,
    ) -> List[int]:
        """填充序列并追加分类token。

        English: Pad to ``max_length`` and append ``cls`` token at tail.
        """
        if len(tokens) > max_length - 1:
            tokens = tokens[:max_length - 1]

        # Add padding to reach max_length-1
        if len(tokens) < max_length - 1:
            tokens = tokens + [pad_token_id] * (max_length - 1 - len(tokens))

        # Add class token at the end
        tokens = tokens + [cls_token_id]
        return tokens


class SigLipTokenizer:
    """SigLIP兼容SentencePiece词表的包装器。

    English: Wrapper for loading SigLIP-friendly SentencePiece tokenizers from
    remote vocabularies.
    """
    VOCAB_FILES = {
        # english, vocab_size=32_000
        "c4-en": "http://storage.googleapis.com/t5-data/vocabs/cc_en.32000/sentencepiece.model",
        # used in multilingual models (mT5, PaLI), vocab_size=250_000
        "mc4": "http://storage.googleapis.com/t5-data/vocabs/mc4.250000.100extra/sentencepiece.model",
        # used in SigLIP2 models, vocab_size=256000
        "gemma": "http://storage.googleapis.com/big_vision/gemma_tokenizer.model",
    }

    def __init__(
            self,
            tokenizer_name: str,
            context_length: Optional[int] = 64,
    ):
        """初始化SigLIP SentencePiece分词器。

        English: Load SentencePiece vocabulary by name or URL for SigLIP models.

        Args:
            tokenizer_name (str): 词表名称或本地路径。
            context_length (Optional[int]): 默认上下文长度。
        """

        if 'gemma' in tokenizer_name:
            from transformers import GemmaTokenizerFast
            tokenizer_cls = partial(
                GemmaTokenizerFast, padding_side='right', add_bos_token=False, add_eos_token=True)
        else:
            from transformers import T5TokenizerFast
            tokenizer_cls = partial(T5TokenizerFast, extra_ids=0)

        if tokenizer_name in self.VOCAB_FILES:
            # FIXME temporary hack?
            import tempfile
            import fsspec
            vocab_file = self.VOCAB_FILES[tokenizer_name]
            with tempfile.NamedTemporaryFile('wb') as dst:
                with fsspec.open(vocab_file, 'rb') as src:
                    dst.write(src.read())
                self.tokenizer = tokenizer_cls(dst.name, legacy=False)
        else:
            self.tokenizer = tokenizer_cls(tokenizer_name, legacy=False)

        self.tokenizer.pad_token_id = 0 if 'gemma' in tokenizer_name else 1
        self.tokenizer.eos_token_id = 1
        self.context_length = context_length

    def save_pretrained(self, dest: str) -> None:
        """保存SentencePiece分词器文件到本地。

        English: Dump tokenizer assets for downstream use.
        """

        self.tokenizer.save_pretrained(dest)

    def __call__(self, texts: Union[str, List[str]], context_length: Optional[int] = None) -> torch.Tensor:
        """执行SigLIP风格分词并返回固定长度张量。

        English: Tokenize inputs with canonical cleaning and SentencePiece model.
        """
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length in class init or call.'

        texts = [canonicalize_text(basic_clean(text)) for text in texts]
        output = self.tokenizer(
            texts,
            return_tensors='pt',
            max_length=context_length,
            padding='max_length',
            truncation=True,
        )
        return output.input_ids
