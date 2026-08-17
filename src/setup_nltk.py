"""
setup_nltk.py
-------------
Script para baixar os recursos NLTK necessários para o projeto.
Execute uma vez após criar o ambiente virtual:

    python src/setup_nltk.py
"""

import nltk


def download_resources():
    resources = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("stemmers/rslp", "rslp"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for path, pkg in resources:
        try:
            nltk.data.find(path)
            print(f"[OK] {pkg} já instalado.")
        except LookupError:
            print(f"[BAIXANDO] {pkg}...")
            nltk.download(pkg)

    print("\nTodos os recursos NLTK estão prontos!")


if __name__ == "__main__":
    download_resources()
