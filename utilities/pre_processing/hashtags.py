from re import compile, findall, subn

# NOTE: The hashtag_regex assumes emojis have already been removed
hashtag_regex = compile(r'#[a-zA-Z0-9_]+')
hashtag_parser_regex = compile(r'[a-z]+|[A-Z][a-z]+|[A-Z]+(?![a-z])|\d+')


def split_hashtags(document, get_header=False):
    """ Identifies hashtags and splits them, where possible """
    if get_header: return 'hashtag_count'

    def replace(_match):
        hashtag = _match.group(0)
        bits = [bit.lower() for bit in findall(hashtag_parser_regex, hashtag)]

        return ' '.join(bits)

    document, num_hashtags = subn(hashtag_regex, replace, document)

    return num_hashtags, document
