import json

with open("notebooks/01_data_exploration.ipynb", "r") as f:
    config = json.load(f)

for cell in config['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if "get_emotion_label" in line:
                line = line.replace("from src.utils import get_emotion_label", "from src.utils import EMOTION_LABELS")
                line = line.replace("get_emotion_label(k)", "EMOTION_LABELS[k]")
                line = line.replace("get_emotion_label(i)", "EMOTION_LABELS[i]")
                cell['source'][i] = line

with open("notebooks/01_data_exploration.ipynb", "w") as f:
    json.dump(config, f, indent=2)
