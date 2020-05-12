from re import compile, sub

# CHECK double check that this characterizes all links seen in other document types
# For twitter documents this should be good..
url_regex = compile(r'http(s)?://(w{3}\.)?(([\w\-_]+\.)+\w{1,6})(/[\w&$\-_.+!*\'()?=#;%:~,]*)*|'
                    r'http:?(/){0,2}\S*$')


def pull_hyperlinks(document, get_header=False):
    """ Locates hyperlinks in document and removes them """
    if get_header: return 'hyperlinks'

    def replace(_match):
        if _match.group(3) != 't.co':
            url = _match.group(3)
            if url is not None:
                urls.add(url)

        return ' url '

    urls = set()
    document = sub(url_regex, replace, document)
    urls = ('[' + ','.join(list(urls)) + ']') if len(urls) > 0 else ''

    return urls, document
