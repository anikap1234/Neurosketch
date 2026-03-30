# =========================
# BASE CODE BUILDER
# =========================

def build_code(layers):
    layer_str = ",\n            ".join(layers)

    return f"""
import torch
import torch.nn as nn

class GeneratedModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            {layer_str}
        )

    def forward(self, x):
        return self.model(x)
"""


# =========================
# DETAILED MODE (GRAPH-DRIVEN)
# =========================

def generate_detailed_code(graph):
    layers = []

    for node in graph["nodes"]:
        text = node["text"].lower()

        if "conv" in text:
            layers.append("nn.Conv2d(3, 64, 3, padding=1)")

        elif "relu" in text:
            layers.append("nn.ReLU()")

        elif "pool" in text:
            layers.append("nn.MaxPool2d(2)")

        elif "flatten" in text:
            layers.append("nn.Flatten()")

        elif "linear" in text or "dense" in text:
            layers.append("nn.Linear(128, 10)")

        elif "lstm" in text:
            layers.append("nn.LSTM(128, 128, batch_first=True)")

    # fallback if OCR fails badly
    if not layers:
        layers = ["nn.Linear(100, 10)"]

    print("\n[CODEGEN] Detailed mode → graph-driven")

    return build_code(layers)


# =========================
# PIPELINE FROM GRAPH
# =========================

def build_pipeline_skeleton(graph):
    steps = []

    for node in graph["nodes"]:
        text = node["text"].lower()

        if "pre" in text:
            steps.append("preprocess")

        elif "feature" in text or "hog" in text:
            steps.append("feature_extraction")

        elif "cam" in text:
            steps.append("postprocess")

    # remove duplicates (preserve order)
    steps = list(dict.fromkeys(steps))

    code = "\n\n# ===== PIPELINE =====\n"

    for step in steps:

        if step == "preprocess":
            code += """
def preprocess(x):
    # TODO: normalization / resizing
    return x
"""

        elif step == "feature_extraction":
            code += """
def feature_extraction(x):
    # TODO: HOG / embeddings
    return x
"""

        elif step == "postprocess":
            code += """
def postprocess(x):
    # TODO: CAM / visualization
    return x
"""

    return code


# =========================
# ABSTRACT MODE (TEMPLATE + PIPELINE)
# =========================

def generate_abstract_code(graph, model_type):
    print("\n[CODEGEN] Abstract mode → template-based")

    # 🔥 TEMPLATE (kept intentionally)
    if model_type == "unet":
        layers = [
            "nn.Conv2d(3, 64, 3, padding=1)",
            "nn.ReLU()",
            "nn.MaxPool2d(2)",
            "nn.ConvTranspose2d(64, 32, 2, stride=2)",
            "nn.Conv2d(32, 1, 1)"
        ]

    elif model_type == "cnn":
        layers = [
            "nn.Conv2d(3, 32, 3)",
            "nn.ReLU()",
            "nn.MaxPool2d(2)",
            "nn.Flatten()",
            "nn.Linear(128, 10)"
        ]

    elif model_type == "rnn":
        layers = [
            "nn.LSTM(128, 128, batch_first=True)",
            "nn.Linear(128, 10)"
        ]

    else:  # MLP fallback
        layers = [
            "nn.Linear(100, 128)",
            "nn.ReLU()",
            "nn.Linear(128, 10)"
        ]

    code = build_code(layers)

    # 🔥 ADD PIPELINE (important)
    code += build_pipeline_skeleton(graph)

    return code


# =========================
# MAIN ENTRY
# =========================

def generate_code(graph, mode, model_type):
    if mode == "detailed":
        return generate_detailed_code(graph)
    else:
        return generate_abstract_code(graph, model_type)