# analyzer.py
from nltk_utils import NLTKUtils

class MessageAnalyzer:
    """
    Classe que analisa uma mensagem de texto, retornando:
    - humor refinado (positivo, neutro, negativo, irritado, etc.)
    - intensidade emocional
    - urgência (com ajustes por canal, tipo de cliente e tempo de abertura)
    - entidades nomeadas (pessoa, organização, local)
    """
    def __init__(self):
        self.nlp = NLTKUtils()
        self.INTENSIFICADORES = ["agora", "imediatamente", "já", "urgente", "urgência"]
        self.AMEACAS = ["cancelar", "trocar fornecedor", "reembolso", "processo"]
        self.IRRITADO = ["furioso", "revoltado", "insuportável", "ódio"]

    def analyze(self, text: str, canal: str = None, tipo_cliente: str = None, dias_aberto: int = 0) -> dict:
        tokens = self.nlp.tokenize(text)
        lemmas = self.nlp.lemmatize(tokens)
        scores = self.nlp.sentiment_scores(text)
        entities = self.nlp.extract_entities(tokens)

        humor, intensidade = self._classify_humor(scores, lemmas)
        urgencia = self._classify_urgency(lemmas, canal, tipo_cliente, dias_aberto)

        return {
            "mensagem": text,
            "humor": humor,
            "intensidade_emocional": intensidade,
            "urgencia": urgencia,
            "entidades": entities
        }

    def _classify_humor(self, scores: dict, lemmas: list[str]) -> tuple[str, str]:
        cmpd = scores['compound']
        # Intensidade emocional
        if abs(cmpd) >= 0.7:
            intensidade = 'extremo'
        elif abs(cmpd) >= 0.5:
            intensidade = 'alto'
        elif abs(cmpd) >= 0.3:
            intensidade = 'moderado'
        else:
            intensidade = 'leve'

        # Definição de humor refinado
        if any(a in lemmas for a in self.AMEACAS) and cmpd < -0.6:
            humor = 'irritado-extremo'
        elif any(i in lemmas for i in self.IRRITADO):
            humor = 'irritado'
        elif cmpd >= 0.5:
            humor = 'positivo'
        elif scores['pos'] > 0:
            humor = 'agradecido'
        elif cmpd > -0.3:
            humor = 'neutro'
        else:
            humor = 'negativo'

        return humor, intensidade

    def _classify_urgency(self, lemmas: list[str], canal: str, tipo_cliente: str, dias_aberto: int) -> str:
        # Pontuação baseada em palavras-chave e intensificadores
        pontos = sum(1 for w in lemmas if w in ["cancelar","erro","falha","atraso"])
        pontos += sum(2 for w in lemmas if w in self.INTENSIFICADORES)

        if any(a in lemmas for a in self.AMEACAS):
            urg = 'alta-extrema'
        elif pontos >= 5:
            urg = 'alta'
        elif pontos >= 3:
            urg = 'média'
        elif pontos >= 1:
            urg = 'baixa'
        else:
            urg = 'nenhuma'

        # Ajustes contextuais: canal, tipo de cliente, tempo de ticket
        if canal in ('chat', 'telefone') and urg != 'alta-extrema':
            urg = self._raise_priority(urg)
        if tipo_cliente in ('VIP', 'premium') and urg != 'alta-extrema':
            urg = self._raise_priority(urg)
        if dias_aberto >= 7 and urg != 'alta-extrema':
            urg = self._raise_priority(urg)

        return urg

    @staticmethod
    def _raise_priority(current: str) -> str:
        mapping = {
            'nenhuma': 'baixa',
            'baixa': 'média',
            'média': 'alta',
            'alta': 'alta'
        }
        return mapping.get(current, current)
