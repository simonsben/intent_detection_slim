from bs4 import BeautifulSoup
from warnings import filterwarnings

# Stop beautiful soup from throwing warnings about url-like text
filterwarnings("ignore", category=UserWarning, module='bs4')


def remove_quotes(document, get_header=False):
    """ Removes quotes from (primarily) storm-front content """
    if get_header: return 'quotes'
    soup = BeautifulSoup(document, 'html.parser')

    quotes = soup.find_all('div', {'style': 'margin:20px; margin-top:5px; '})
    quote_citation = soup.find_all('div', {'align': 'right'})

    count = len(quotes)
    for quote in quotes:
        quote.decompose()
    for citation in quote_citation:
        citation.decompose()

    return count, str(soup)
