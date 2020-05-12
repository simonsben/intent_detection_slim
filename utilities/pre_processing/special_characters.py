from html import unescape
from unidecode import unidecode


def manage_special_characters(document, get_header=False):
    if get_header: return None

    # Convert html characters to string equivalent
    document = unescape(document).replace('’', '\'')

    # Convert unicode characters
    document = unidecode(document)

    return None, document
