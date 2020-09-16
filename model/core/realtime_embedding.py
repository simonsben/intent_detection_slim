from fasttext.FastText import _FastText
from tensorflow.keras.utils import Sequence
from numpy import zeros, ones, ndarray, abs
from config import batch_size, max_tokens
from math import ceil


class RealtimeEmbedding(Sequence):
    """ Extends TensorFlow Sequence to provide on-the-fly fastText token embedding """
    def __init__(self, embedding_model, data_source, labels=None, uniform_weights=False):
        """
        Implements Keras data sequence for on-the-fly embedding generation

        :param _FastText embedding_model: FastText embedding model
        :param ndarray data_source: List of documents to embed on the fly
        :param ndarray labels: Array of data labels
        :param bool labels_in_progress: Whether passed labels should be taken as initial labels and marked
        :param bool uniform_weights: Whether weights should be uniform (i.e. 1)
        """

        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_model.get_dimension()
        self.embedding_cache = {}

        self.data_source = data_source
        self.working_data_source = self.data_source

        self.labels = labels
        self.working_labels = self.labels

        self.working_mask = None
        self.is_training = False

        self.concrete_weight = 1
        self.midpoint = 0.5
        self.uniform_weights = uniform_weights
        self.data_length = ceil(len(self.working_data_source) / batch_size)

    def update_labels(self, new_labels):
        """ Updates the labels being fed """
        self.labels = new_labels.copy()
        self.set_mask(self.working_mask)

    # TODO add line to automatically mask uncertain values when training
    def set_usage_mode(self, is_training):
        """ Changes usage mode """
        if is_training is True and self.labels is None:
            raise AttributeError('Cannot use in training mode if there are no labels.')

        self.is_training = is_training

    def set_mask(self, definite_mask):
        """ Updates the current mask being applied to the data """
        self.working_mask = definite_mask

        if self.working_mask is not None:   # If not None, apply mask to data
            self.working_data_source = self.data_source[self.working_mask]
            self.working_labels = self.labels[self.working_mask]

        # If updated mask is None, make working set entire set
        else:
            self.working_data_source = self.data_source
            self.working_labels = self.labels

        # Recompute data length
        self.data_length = ceil(len(self.working_data_source) / batch_size)

    def get_sample_weights(self, batch_start, batch_end):
        """
        Returns sample weights for data samples.
        Weights are computed using the function w = 2(x - .5) when x = (.5, 1], and the negation when x = [0, .5)
        """
        if self.uniform_weights:
            return ones(batch_end - batch_start)

        labels = self.working_labels[batch_start:batch_end]
        weights = compute_sample_weights(labels, self.midpoint)

        return weights

    def embed_data(self, data_subset):
        """ Computes word embeddings for provided data subset """
        # Initialize embedding of data
        embedded_data = zeros((data_subset.shape[0], max_tokens, self.embedding_dimension), float)

        # Embed all documents
        for doc_index, document in enumerate(data_subset):
            document_tokens = document.split(' ')[:max_tokens]  # Split document into tokens and limit

            # For each token in document
            for token_index, token in enumerate(document_tokens):
                # If token embedding is not already cached, compute it and store
                if token not in self.embedding_cache:
                    self.embedding_cache[token] = self.embedding_model.get_word_vector(token)

                # Add embedding to array
                embedded_data[doc_index, token_index] = self.embedding_cache[token]

        return embedded_data

    def __len__(self):
        """ Overrides length method to compute the length in batches """
        if self.is_training:
            return self.data_length

        return ceil(len(self.data_source) / batch_size)

    def __getitem__(self, index):
        """ Provides the batch of data at a given index """
        batch_start = int(index * batch_size)
        batch_end = batch_start + batch_size

        # Get batch of data
        source = self.working_data_source if self.is_training else self.data_source
        working_data = source[batch_start:batch_end]

        # Correct batch end to reflect current atch index
        if batch_end > len(source):
            batch_end = len(source)

        embedded_data = self.embed_data(working_data)

        # If training also return labels
        if self.is_training:
            # Get batch labels and convert to boolean
            label_subset = self.working_labels[batch_start:batch_end]
            label_subset = label_subset > self.midpoint

            loss_weights = self.get_sample_weights(batch_start, batch_end)

            return embedded_data, label_subset, loss_weights
        return embedded_data


def compute_sample_weights(labels, midpoint=0.5):
    """
    Computes sample weights for training

    :param ndarray labels: Array of current labels
    :param float midpoint: Midpoint for computing the loss weight around
    """

    return 2 * abs(labels - midpoint)
