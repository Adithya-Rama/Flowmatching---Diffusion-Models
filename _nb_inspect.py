import json, os
nb_path = os.path.join(os.path.dirname(__file__), "COMP8650_FlowMatching.ipynb")
with open(nb_path, encoding="utf-8") as fh:
    nb = json.load(fh)
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "train_meanflow" in src or "MF_RUN_TAG" in src or "MF_STEP_COUNTS" in src:
        print(f"Cell {i} ({cell['cell_type']}):")
        print(src[:1200])
        print("---")
