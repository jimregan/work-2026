1 from datasets import ClassLabel
         2
         3 if LOAD_FROM_HF:
         4     ds = load_dataset(DATASET_PATH)
         5 else:
         6     ds = load_from_disk(DATASET_PATH)
         7
         8 if "train" not in ds:
         9     if not hasattr(ds.features.get("dialect_label"), "names"
           ):
        10         unique_dialects = sorted(set(ds\["dialect_label"\]))
        11         ds = ds.cast_column("dialect_label", ClassLabel(name
           s=unique_dialects))
        12     ds = ds.train_test_split(test_size=0.1, seed=SEED, strat
           ify_by_column="dialect_label")
        13
        14 print(ds)
        15 print()
        16 from collections import Counter
        17 names = ds\["train"\].features\["dialect_label"\].names
        18 print("Train dialect distribution:")
        19 for idx, count in sorted(Counter(ds\["train"\]\["dialect_label"
           \]).items()):
        20     print(f"  {names\[idx\]:<20} {count}")



        1 LABELS   = ds\["train"\].features\["dialect_label"\].nam
          es
        2 label2id = {l: i for i, l in enumerate(LABELS)}
        3 id2label = dict(enumerate(LABELS))
        4 print("Labels:", LABELS)
        5                                                                       
        6 ds = ds.map(lambda x: {"label": x\["dialect_label"\]},
           batched=True)