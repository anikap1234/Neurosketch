from detect import detect_boxes
from ocr import extract_texts
from graph import build_graph

from pipeline_builder import graph_to_pipeline
from architecture_builder import build_architecture
from generator import generate_code


def run(image_path):
    print("\n--- STAGE 1: DETECTION ---")
    img, boxes = detect_boxes(image_path)
    print("Boxes:", boxes)

    print("\n--- STAGE 2: OCR ---")
    nodes = extract_texts(img, boxes)

    print("\n--- STAGE 3: GRAPH ---")
    graph = build_graph(nodes)

    print("\n--- STAGE 4: PIPELINE BUILD ---")
    pipeline = graph_to_pipeline(graph)

    print("\n--- STAGE 5: ARCHITECTURE BUILD ---")
    architecture = build_architecture(pipeline)

    print("\n--- STAGE 6: CODE GENERATION ---")
    code = generate_code(architecture)

    with open("generated_model.py", "w") as f:
        f.write(code)

    print("\n✅ DONE → generated_model.py created")


if __name__ == "__main__":
    run(r"C:\Anika\Projects\neurosketch\images\sample1.png")