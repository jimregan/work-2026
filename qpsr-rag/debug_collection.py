import chromadb

client = chromadb.PersistentClient(path="/data/chroma_db")
print("collections:", client.list_collections())
print("count:", client.get_collection("stl-qpsr").count())
