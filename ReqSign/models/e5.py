from sentence_transformers import SentenceTransformer


def get_similarity_e5(req, pattern):

    model = SentenceTransformer("intfloat/e5-large-v2")

    req_emb = model.encode(req)
    pattern_emb = model.encode(pattern)

    similarities = model.similarity(req_emb, pattern_emb)
    return similarities