from langchain_community.embeddings import HuggingFaceHubEmbeddings
import inspect

print("Inspecting HuggingFaceHubEmbeddings...")
sig = inspect.signature(HuggingFaceHubEmbeddings.__init__)
print(f"Constructor signature: {sig}")

# Try to initialize with a test
try:
    # Note: we won't actually call it without a token, just testing import and signature
    print("HuggingFaceHubEmbeddings is available.")
except Exception as e:
    print(f"Error: {e}")
