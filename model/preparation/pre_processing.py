from multiprocessing import Pool
from functools import partial
from config import n_threads
from utilities import load_data
from model.preparation.contexts import split_into_contexts
from utilities import save_contexts
from utilities.pre_processing import *

# Default set and ordering of pre-processing functions
standard_processes = [
    original_length,
    remove_quotes,
    manage_special_characters,
    pull_hyperlinks,
    count_tags,
    count_images,
    count_bracket_text,
    count_emojis,
    split_hashtags,
    count_upper,
    count_acronym,
    count_digits,
    count_repeat_instances,
    run_partial_clean,
]


def apply_process(document, processes):
    """
    Applies the pre-processing filters to a document

    :param str document: Contents of a document
    :param list processes: A list of processes to be applied
    :return str: Processed document content
    """

    # For each pre-processing step to be applied
    for process in processes:
        _, document = process(document if isinstance(document, str) else '')

    return document


def process_documents(source_path, target_path, processes=None, content_index=0):
    """
    Pre-processes all documents within a CSV file.

    :param Path source_path: Filename for the source CSV file
    :param Path target_path: Filename for destination file
    :param list processes: List of pre-processing functions, (document_content) -> (value, modified_content)
    :param int content_index: Index of the document content
    """
    data = load_data(source_path, index_col=None).values[:, content_index]

    if processes is None:
        processes = standard_processes
    processor = partial(apply_process, processes=processes)

    workers = Pool(n_threads)                   # Define workers
    documents = workers.map(processor, data)   # Apply processing
    workers.close()                             # Close document queue
    workers.join()                              # Wait for processes to finish

    contexts, indexes = split_into_contexts(documents)
    save_contexts(contexts, indexes, target_path)
