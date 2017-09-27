
try:
    from typing import NamedTuple
    Seq2SeqConfig = NamedTuple('Seq2SeqParams', (
        ('message_len', int),
        ('batch_size', int),
        ('context_size', int),
        ('embed_size', int),
        ('use_cuda', bool),
        ('vocab_size', int),
    ))
except ImportError:
    from collections import namedtuple
    Seq2SeqConfig = namedtuple('Seq2SeqParams', (
        'message_len',
        'batch_size',
        'context_size',
        'embed_size',
        'use_cuda',
        'vocab_size',
    ))

