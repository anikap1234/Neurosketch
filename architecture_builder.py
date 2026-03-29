def build_architecture(pipeline):
    architecture = {
        "type": "mlp",
        "layers": []
    }

    for step in pipeline:
        if step["type"] == "model":
            architecture["type"] = step["name"]

    if architecture["type"] == "unet":
        architecture["layers"] = [
            "conv", "relu", "pool",
            "conv", "relu",
            "upconv", "relu",
            "conv"
        ]

    elif architecture["type"] == "cnn":
        architecture["layers"] = [
            "conv", "relu", "pool",
            "conv", "relu",
            "flatten",
            "dense"
        ]

    else:
        architecture["layers"] = [
            "dense", "relu", "dense"
        ]

    print("\nARCHITECTURE:")
    print(architecture)

    return architecture