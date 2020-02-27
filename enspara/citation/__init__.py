import atexit

from . import citation
from .citation import cite


atexit.register(citation.citation_printer)
citation.add_citation('enspara')
