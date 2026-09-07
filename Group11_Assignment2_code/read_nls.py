import json
for h in [2, 4, 8, 16, 32]:
    with open(f"outputs/classification/nls/h{h}/metrics_val.json") as f:
        m = json.load(f)
    print(f"h={h}x{h}")
    print(f"Prec: {[round(x,4) for x in m['precisions']]}")
    print(f"Rec : {[round(x,4) for x in m['recalls']]}")
    print(f"F1  : {[round(x,4) for x in m['f1_scores']]}")
    print()
