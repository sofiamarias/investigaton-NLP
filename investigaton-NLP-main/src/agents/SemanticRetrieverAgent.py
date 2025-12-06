from src.datasets.LongMemEvalDataset import LongMemEvalInstance
"""
Este agente se encarga de, dada una instancia del benchmark, con su query y sus sessions,
conseguir los top-k mensajes más relevantes.(Pensamos en filtrar primero por sesiones)
"""
class SemanticRetrieverAgent:
    def __init__(self, model, embedding_model_name):
        self.model = model
        self.embedding_model_name = embedding_model_name
    
    def answer(self, instance: LongMemEvalInstance):
        pass
