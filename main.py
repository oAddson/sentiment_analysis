from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from analyzer import MessageAnalyzer
from typing import List


app = FastAPI(
    title="API Avançada de Análise de Mensagens",
    description="Classifica humor, urgência e extrai entidades de texto usando NLTK",
    version="1.0.0"
)

analyzer = MessageAnalyzer()

class MessageRequest(BaseModel):
    texto: str = Field(..., example="Estou muito frustrado com o serviço!")
    canal: str | None = Field(None, example="chat")
    tipo_cliente: str | None = Field(None, example="VIP")
    dias_aberto: int = Field(0, ge=0, example=5)

class MessageResponse(BaseModel):
    mensagem: str
    humor: str
    intensidade_emocional: str
    urgencia: str
    entidades: list[tuple[str, str]]

@app.post("/analisar/", response_model=MessageResponse)
def analisar(request: MessageRequest):
    if not request.texto or not request.texto.strip():
        raise HTTPException(status_code=400, detail="O texto da mensagem não pode estar vazio.")
    result = analyzer.analyze(
        text=request.texto,
        canal=request.canal,
        tipo_cliente=request.tipo_cliente,
        dias_aberto=request.dias_aberto
    )
    return result
class BatchMessageRequest(BaseModel):
    mensagens: List[MessageRequest]

class BatchMessageResponse(BaseModel):
    resultados: List[MessageResponse]

@app.post("/analisar/batch/", response_model=BatchMessageResponse)
def analisar_em_batch(request: BatchMessageRequest):
    resultados = []

    for msg in request.mensagens:
        if not msg.texto or not msg.texto.strip():
            raise HTTPException(status_code=400, detail="Uma das mensagens está vazia.")
        resultado = analyzer.analyze(
            text=msg.texto,
            canal=msg.canal,
            tipo_cliente=msg.tipo_cliente,
            dias_aberto=msg.dias_aberto
        )
        resultados.append(resultado)

    return {"resultados": resultados}
