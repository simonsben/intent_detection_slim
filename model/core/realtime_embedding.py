from fasttext.FastText import _FastText
from tensorflow.keras.utils import Sequence
from numpy import zeros
from config import batch_size as b_size, max_tokens


class RealtimeEmbedding(Sequence):
    """ Extends TensorFlow Sequence to provide on-the-fly fastText token embedding """
    def __init__(self, embedding_model, data_source, labels=None, batch_size=b_size):
        """
        Implements Keras data sequence for on-the-fly embedding generation

        :param _FastText embedding_model: FastText embedding model
        :param list data_source: List of documents to embed on the fly
        :param ndarray labels: Array of data labels
        :param int batch_size: Batch size when documents are requested
        """

        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_model.get_dimension()
        self.embedding_cache = {}

        self.raw_data_source = data_source
        self.data_source = self.raw_data_source

        self.raw_labels = labels
        self.labels = self.raw_labels

        self.mask = None
        self.is_training = False
        self.batch_size = batch_size
        self.data_length = int(len(self.data_source) / self.batch_size) + 1

    def update_labels(self, new_labels):
        """ Updates the labels being fed """
        self.labels = new_labels.copy()

    def set_usage_mode(self, is_training):
        """ Changes usage mode """
        if is_training is True and self.raw_labels is None:
            raise AttributeError('Cannot use in training mode if there are no labels.')

        self.is_training = is_training

    def set_mask(self, mask):
        """ Updates the current mask being applied to the data """
        self.mask = mask
        if self.mask is not None:
            self.data_source = self.raw_data_source[self.mask]
            self.labels = self.raw_labels[self.mask]
        else:
            self.data_source = self.raw_data_source
            self.labels = self.labels

        self.data_length = int(len(self.data_source) / self.batch_size) + 1

    def __len__(self):
        """ Overrides length method """
        if self.is_training:
            return self.data_length
        return int(len(self.raw_data_source) / self.batch_size) + 1

    def __getitem__(self, index):
        """ Provides the batch of data at a given index """
        batch_start = int(index * self.batch_size)
        batch_end = batch_start + self.batch_size

        # Get batch of data
        source = self.data_source if self.is_training else self.raw_data_source
        data_subset = source[batch_start:batch_end]

        # Initialize embedding of data
        embedded_data = zeros((data_subset.shape[0], max_tokens, self.embedding_dimension), float)

        # Embed all documents
        for doc_index, document in enumerate(data_subset):
            document_tokens = document.split(' ')[:max_tokens]      # Split document into tokens and limit

            # For each token in document
            for token_index, token in enumerate(document_tokens):
                # If token embedding is not already cached, compute it and store
                if token not in self.embedding_cache:
                    self.embedding_cache[token] = self.embedding_model.get_word_vector(token)

                # Add embedding to array
                embedded_data[doc_index, token_index] = self.embedding_cache[token]

        # If training also return labels
        if self.is_training:
            label_subset = self.labels[batch_start:batch_end]
            return embedded_data, label_subset
        return embedded_data
