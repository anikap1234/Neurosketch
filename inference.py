def normalize(text):
    text = text.replace("modal", "model")
    text = text.replace("feadusa", "feature")
    text = text.replace("proce", "process")
    text = text.replace("fs0", "pre")
    return text


def extract_model_nodes(nodes):
    model_nodes = []

    # 1️⃣ Try text-based detection
    for n in nodes:
        text = normalize(n["text"]).lower()

        if any(k in text for k in ["unet", "onet", "cnn", "lstm"]):
            model_nodes.append(n)

    # 2️⃣ POSITION fallback
    if len(model_nodes) == 0 and len(nodes) > 0:
        sorted_nodes = sorted(nodes, key=lambda x: x["center"][1])
        mid_idx = len(sorted_nodes) // 2

        model_nodes = [sorted_nodes[mid_idx]]

        print("⚠️ Using POSITION fallback for model detection")

    return model_nodes


def classify_mode(nodes):
    texts = [normalize(n["text"]) for n in nodes]

    keywords = ["conv", "relu", "dense", "pool", "flatten"]
    layer_nodes = [t for t in texts if any(k in t for k in keywords)]

    if len(layer_nodes) >= 2:
        return "detailed"
    return "abstract"


def classify_type(nodes):
    model_nodes = extract_model_nodes(nodes)

    print("\nMODEL NODES:", [n["text"] for n in model_nodes])

    text = " ".join([normalize(n["text"]) for n in model_nodes]).lower()

    if "unet" in text or "onet" in text:
        return "unet"
    elif "conv" in text or "cnn" in text:
        return "cnn"
    elif "lstm" in text or "rnn" in text:
        return "rnn"
    else:
        return "unet"  # fallback