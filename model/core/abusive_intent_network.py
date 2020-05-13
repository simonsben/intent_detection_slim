from keras.models import Model
from config import execute_verbosity
from numpy import ndarray, vectorize, cumsum, sqrt, argmin, histogram


def compute_inner_product(value_one, value_two, norm=2):
    """ Vector norm, euclidean inner product by default """
    return (value_one ** norm + value_two ** norm) ** (1 / norm)


# TODO correct to space bins based on data distribution (similar to a lebesgue integral)
def estimate_cumulative(data, num_bins=150):
    """
    Estimates the cumulative distribution of a dataset

    :param data: Data vector, numpy array
    :param num_bins: Number of bins to use in the estimation, int
    :return: Estimated function
    """
    distribution, bin_edges = histogram(data, bins=num_bins)
    bin_edges = bin_edges[:-1]

    cumulative = cumsum(distribution)
    cumulative -= cumulative[0]
    cumulative = sqrt(cumulative / cumulative[-1])

    def cumulative_function(prediction):
        relative_locations = bin_edges <= prediction
        if relative_locations[-1]:              # If its in the last bin
            return cumulative[-1]

        bin_index = argmin(relative_locations)  # Get index of the bin
        return cumulative[bin_index]            # Return approx cumulative sum at the point

    return cumulative_function


def compute_abusive_intent(intent_predictions, abuse_predictions, method='product'):
    """
    Compute a 'score' for abusive intent from intent and abuse predictions

    :param ndarray intent_predictions: Array of intent predictions
    :param ndarray abuse_predictions: Array of abuse predictions
    :param str method: Abusive intent calculation method to use [default product]
    :return ndarray: Array of predictions same shape as input arrays
    """
    if not isinstance(intent_predictions, ndarray):
        raise TypeError('Expected intent predictions to be a numpy array.')
    if not isinstance(abuse_predictions, ndarray):
        raise TypeError('Expected abuse predictions to be a numpy array.')
    if intent_predictions.shape != abuse_predictions.shape:
        raise TypeError('Intent predictions and abuse predictions must be the same length.')
    if len(intent_predictions.shape) > 1:
        raise TypeError('Predictions should be a vector, not an array')

    if method == 'cdf':
        cumulative_function = vectorize(estimate_cumulative(intent_predictions))
        return abuse_predictions * cumulative_function(intent_predictions)
    elif method == 'euclidean':
        norm = vectorize(compute_inner_product)
        return norm(intent_predictions, abuse_predictions)
    elif method == 'product':
        pass
    else:
        UserWarning('Invalid method choice, using product')

    return intent_predictions * abuse_predictions


def predict_abusive_intent(realtime_documents, abusive_intent_network, method='product'):
    """
    Makes abusive intent predictions for a list of pre-processed documents

    :param RealtimeEmbedding realtime_documents: list or array of pre-processed documents
    :param Model abusive_intent_network: keras network trained to predict abuse and intent
    :param str method: method used to make abusive intent predictions
    :return tuple: tuple of abuse, intent, and abusive-intent predictions
    """
    abuse_predictions, intent_predictions = [
        predictions.reshape(-1) for predictions in abusive_intent_network.predict_generator(
            realtime_documents, verbose=execute_verbosity
        )
     ]

    abusive_intent_predictions = compute_abusive_intent(intent_predictions, abuse_predictions, method)

    return abuse_predictions, intent_predictions, abusive_intent_predictions
