import langchain_huggingface
import langchain_community
print("langchain_huggingface versions and members:")
try:
    from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceInferenceAPIEmbeddings
    print("Successfully imported HuggingFaceInferenceAPIEmbeddings")
except ImportError as e:
    print(f"Failed to import from langchain_huggingface: {e}")

print("\nTrying to find alternative embedding classes...")
try:
    from langchain_community.embeddings import HuggingFaceHubEmbeddings
    print("Found HuggingFaceHubEmbeddings in langchain_community")
except ImportError:
    print("HuggingFaceHubEmbeddings not found in langchain_community")
