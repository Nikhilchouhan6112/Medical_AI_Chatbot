import langchain_huggingface.embeddings as hf_embeddings
import inspect

print("Members of langchain_huggingface.embeddings:")
for name, obj in inspect.getmembers(hf_embeddings):
    if inspect.isclass(obj):
        print(f"Class: {name}")

print("\nChecking huggingface_hub version:")
import huggingface_hub
print(f"huggingface_hub version: {huggingface_hub.__version__}")
