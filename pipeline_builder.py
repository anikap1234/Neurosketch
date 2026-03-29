def normalize(text):
    text = text.lower()
    text = text.replace("feadusa", "feature")
    text = text.replace("proce", "process")
    text = text.replace("modal", "model")
    return text


def classify_node(text):
    text = normalize(text)

    if "input" in text:
        return {"type": "input"}

    elif "preprocess" in text or "process" in text:
        return {"type": "preprocess"}

    elif "feature" in text or "hog" in text:
        return {"type": "feature"}

    elif "unet" in text:
        return {"type": "model", "name": "unet"}

    elif "cnn" in text or "conv" in text:
        return {"type": "model", "name": "cnn"}

    elif "lstm" in text or "rnn" in text:
        return {"type": "model", "name": "rnn"}

    elif "cam" in text:
        return {"type": "postprocess", "name": "cam"}

    elif "output" in text:
        return {"type": "output"}

    else:
        return {"type": "unknown", "raw": text}


def graph_to_pipeline(graph):
    pipeline = []

    for node in graph["nodes"]:
        step = classify_node(node["text"])

        # ignore useless OCR noise
        if step["type"] != "unknown":
            pipeline.append(step)

    print("\nPIPELINE:")
    for p in pipeline:
        print(p)

    return pipeline