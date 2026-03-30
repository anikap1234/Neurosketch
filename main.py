from detect import detect_boxes
from ocr import extract_texts
from graph import build_graph

from decision import make_decision
from codegen import generate_code


def run(image_path):
    print("\n--- STAGE 1: DETECTION ---")
    img, boxes = detect_boxes(image_path)

    print("\n--- STAGE 2: OCR ---")
    nodes = extract_texts(img, boxes)

    print("\n--- STAGE 3: GRAPH ---")
    graph = build_graph(nodes)
    print(graph)

    print("\n--- STAGE 4: DECISION ---")
    mode, model_type = make_decision(graph)

    print("\n--- STAGE 5: CODE GENERATION ---")
    code = generate_code(graph, mode, model_type)

    with open("generated_model.py", "w") as f:
        f.write(code)

    print("\n✅ DONE → generated_model.py created")


if __name__ == "__main__":
    run(r"C:\Anika\Projects\neurosketch\images\sample1.png")