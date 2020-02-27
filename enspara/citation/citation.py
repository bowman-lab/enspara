import os
import json
import functools

from ..exception import ImproperlyConfigured


def load_citation_db():
    db = set()
    with open(os.path.join(os.path.dirname(__file__), 'articles.json')) as f:
        db = json.load(f)

    for v in db.values():
        assert 'doi' in v
        assert 'title' in v
        assert 'year' in v
        assert len(v['authors']) > 1

    return db


def citation_printer():
    s = ("Thanks for using enspara! Please read "
         "and cite the folllowing articles:\n")

    for k in USED_CITATIONS:
        s += str(CITATION_DB[k]) + '\n'

    print(s)


def add_citation(citekey):
    if citekey not in CITATION_DB:
        raise ImproperlyConfigured(
            "Cannot cite %s, wasn't in citation db:\n%s" %
            (citekey, list(CITATION_DB.keys())))
    USED_CITATIONS.add(citekey)


def cite(citekey):
    def cite_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            add_citation(citekey)
            return func(*args, **kwargs)
        return wrapper
    return cite_decorator


CITATION_DB = load_citation_db()
USED_CITATIONS = set()
