class AudioTooShort(ValueError):
    """Audio inutilisable (trop court). Faute du client -> HTTP 422."""

class ModelContractError(Exception):
    """Le preprocessing ne respecte pas le contrat du modèle (meta.json / ONNX). Bug serveur -> HTTP 500."""