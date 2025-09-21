from engine.services.embedding import EmbeddingService

service = EmbeddingService()
result = service.get_embedding('test')
print(f'Embedding dimension: {len(result["embeddings"])}')
print(f'First 10 values: {result["embeddings"][:10]}')