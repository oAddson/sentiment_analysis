# nltk_utils.py

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 🔧 Baixar recursos necessários para evitar erros em runtime
for pkg in [
    'vader_lexicon',
    'punkt_tab',
    'stopwords',
    'wordnet',
    'averaged_perceptron_tagger_eng',
    'maxent_ne_chunker_tab',
    'words'
]:
    nltk.download(pkg, quiet=True)


class NLTKUtils:
    """
    Classe utilitária que encapsula todas as operações básicas usando NLTK:
    """
    def __init__(self):
        self._init_lemmatizer()
        self._init_stopwords()
        self._init_sentiment()

    def _init_lemmatizer(self):
        self.lemmatizer = WordNetLemmatizer()

    def _init_stopwords(self):
        # Carrega stopwords em português e inglês
        self.stop_words = set(stopwords.words('portuguese')) | set(stopwords.words('english'))

    def _init_sentiment(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def tokenize(self, text: str) -> list[str]:
        """
        Divide o texto em tokens (palavras e pontuação).
        """
        return word_tokenize(text)

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """
        Remove stopwords e transforma tokens em suas formas raiz (verbos).
        """
        lemmas = []
        for token in tokens:
            token_lower = token.lower()
            if token_lower.isalpha() and token_lower not in self.stop_words:
                lemma = self.lemmatizer.lemmatize(token_lower, pos='v')
                lemmas.append(lemma)
        return lemmas

    def pos_tag(self, tokens: list[str]) -> list[tuple[str, str]]:
        """
        Retorna a lista de tokens com suas respectivas classes gramaticais (POS).
        """
        return pos_tag(tokens)

    def extract_entities(self, tokens: list[str]) -> list[tuple[str, str]]:
        """
        Extrai entidades nomeadas (pessoa, organização, local) usando POS + chunking.
        """
        tagged = self.pos_tag(tokens)
        tree = ne_chunk(tagged)
        entities = []
        for subtree in tree:
            if hasattr(subtree, 'label'):
                entity_text = ' '.join(word for word, _ in subtree)
                entities.append((subtree.label(), entity_text))
        return entities

    def sentiment_scores(self, text: str) -> dict[str, float]:
        """
        Retorna as pontuações de sentimento pelo VADER, incluindo 'compound'.
        """
        return self.sentiment_analyzer.polarity_scores(text)
