import langchain_huggingface
import pkgutil

print(f"langchain_huggingface version: {langchain_huggingface.__version__ if hasattr(langchain_huggingface, '__version__') else 'unknown'}")
print("\nModules in langchain_huggingface:")
for loader, module_name, is_pkg in pkgutil.walk_packages(langchain_huggingface.__path__):
    print(module_name)

print("\nTrying to import various embedding classes...")
try:
    from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings
    print("SUCCESS: Imported HuggingFaceInferenceAPIEmbeddings")
except ImportError as e:
    print(f"FAILED: HuggingFaceInferenceAPIEmbeddings - {e}")

try:
    from langchain_huggingface.embeddings import HuggingFaceInferenceAPIEmbeddings
    print("SUCCESS: Imported from langchain_huggingface.embeddings")
except ImportError as e:
    print(f"FAILED: langchain_huggingface.embeddings - {e}")
