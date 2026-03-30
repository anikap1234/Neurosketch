def normalize(text):
    return text.lower()


def is_detailed(nodes):
    keywords = ["conv", "relu", "pool", "linear", "lstm", "rnn"]

    for n in nodes:
        t = normalize(n["text"])
        if any(k in t for k in keywords):
            return True

    return False


def detect_model_type(nodes):
    for n in nodes:
        t = normalize(n["text"])

        if "unet" in t:
            return "unet"
        if "cnn" in t:
            return "cnn"
        if "lstm" in t or "rnn" in t:
            return "rnn"

    return "mlp"


def make_decision(graph):
    nodes = graph["nodes"]

    mode = "detailed" if is_detailed(nodes) else "abstract"
    model_type = detect_model_type(nodes)

    print("\nDECISION:")
    print("Mode:", mode)
    print("Model:", model_type)

    return mode, model_type