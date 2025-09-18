from dotenv import load_dotenv
import os
load_dotenv()

class ServiceSettings:
    IMAGE_PROCESSOR_API_KEY = os.getenv("API_NINJAS_KEY")
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    NLPCLOUD_TOKEN = os.getenv("NLPCLOUD_TOKEN")

class MilvusSettings:
    # Docker-compatible Milvus configuration
    HOST = os.getenv("MILVUS_HOST", "localhost")
    PORT = int(os.getenv("MILVUS_PORT", "19531"))
    COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "rag_documents")
    
    @classmethod
    def get_connection_args(cls):
        """Get connection arguments for Milvus"""
        return {
            "host": cls.HOST,
            "port": cls.PORT
        }
