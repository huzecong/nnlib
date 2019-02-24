from pathlib import Path
from typing import List, Union, overload

import numpy as np

from ..data.dataloader import Vocabulary
from ..torch import *

__all__ = ['Embedding']


class Embedding(nn.Embedding):
    r"""
    Wrapper over :py:class:`nn.Embedding`.

    - Uses ``gensim`` to load embeddings, so simply specify embedding type and file path.
    - Accepts :py:class:`nn.utils.rnn.PackedSequence` as forward input.
    - Support embedding dropout.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, *, embedding_dropout: float = 0.0, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        if embedding_dropout > 0.0:
            self.drop_emb = nn.Dropout(embedding_dropout)  # TODO: Implement correct embedding dropout
        else:
            self.drop_emb = lambda x: x  # identity

    @overload
    def forward(self, input: torch.LongTensor) -> Tensor:
        ...

    @overload
    def forward(self, input: PackedSequence) -> PackedSequence:
        ...

    def forward(self, input):
        r"""
        :param input: A :class:`~torch.LongTensor`, or a :class:`~torch.utils.rnn.PackedSequence`.
        :return: A :class:`~torch.Tensor`, or a :class:`~torch.utils.rnn.PackedSequence`.
        """
        device = self.weight.device
        indices = input.data if isinstance(input, PackedSequence) else input

        # Check that all indices are within range, otherwise CUDA complains with a random cryptic error message
        invalid = (indices < 0) | (indices >= self.num_embeddings)
        if torch.any(invalid).item():
            raise ValueError(f"Lookup index ({indices[invalid][0]}) out of range [0, {self.num_embeddings - 1}]")

        if indices.device != device:
            indices = indices.to(device=device)
        embed = self.drop_emb(super().forward(indices))

        if isinstance(input, PackedSequence):
            packed = nn.utils.rnn.PackedSequence(embed, input.batch_sizes)
            packed.device = device
            return packed
        else:
            return embed

    def __getitem__(self, item: Union[int, List[int]]) -> Tensor:
        r"""
        A convenience wrapper for native Python types.
        """
        if isinstance(item, int):
            item = [item]
        item = torch.tensor(item, dtype=torch.long, device=self.weight.device)
        return self.forward(item)

    def reset_parameters(self) -> None:
        r"""
        Default initialization in PyTorch uses unit normal distribution. Here we simply use uniform.
        """
        self.weight.data.uniform_(-0.1, 0.1)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    # noinspection PyMethodOverriding
    @classmethod
    def from_pretrained(cls, embed_type: str, embed_path: Path, word_vocab: Vocabulary, embed_dim: int,
                        freeze=True, sparse=False) -> 'Embedding':
        r"""
        Creates an :class:`Embedding` instance from external pretrained embeddings.

        :param embed_type: Type of the embedding, can be ``word2vec`` or ``fasttext``.
        :param embed_path: Path to the embedding file.
        :param word_vocab: A vocabulary mapping words to indices.
        :param embed_dim: Dimension of embeddings.
        :param freeze: If ``True``, embeddings are fixed during training.
        :param sparse: If ``True``, sparse embeddings are used. See PyTorch documentation for details.
        """
        embed_path_str = str(embed_path.resolve())
        if embed_type == 'word2vec':
            from gensim.models.word2vec import Word2VecKeyedVectors
            model = Word2VecKeyedVectors.load(embed_path_str)
        elif embed_type == 'fasttext':
            from gensim.models.fasttext import FastTextKeyedVectors
            model = FastTextKeyedVectors.load(embed_path_str)
        elif embed_type == 'glove':
            raise NotImplementedError
        else:
            raise ValueError(f"Embedding type {embed_type} not supported.")

        assert model.vector_size == embed_dim
        embeddings = np.zeros((len(word_vocab), embed_dim))
        for word, idx in word_vocab.items():
            embeddings[idx] = model.get_vector(word)

        embedding = super().from_pretrained(embeddings, freeze=freeze, sparse=sparse)
        # no point in doing the following: `cls` in classmethod points to the subclass
        # embedding.forward = types.MethodType(cls.forward, embedding)
        return embedding
