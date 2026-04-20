# 🧠 NeuroSketch  
### Diagram-Driven Machine Learning Code Synthesis

> Convert hand-drawn ML architecture diagrams + text specifications into executable PyTorch/Keras code using multimodal AI.

---

## 🚀 Overview

NeuroSketch is an end-to-end intelligent system that bridges the gap between **visual ML design and implementation**.  
It allows users to sketch machine learning pipelines (on paper or digitally), provide optional textual specifications, and automatically generate **runnable deep learning code**.

Unlike traditional AutoML systems, NeuroSketch emphasizes **user-driven design**, enabling better understanding, flexibility, and rapid prototyping.

---

## 🎯 Problem Statement

Designing ML models often starts with diagrams, but converting them into code is:
- ❌ Manual and time-consuming  
- ❌ Error-prone  
- ❌ Disconnected from the intuitive design process  

Existing tools:
- Focus on dataset → model (AutoML)
- Do not interpret **hand-drawn architectures**

👉 NeuroSketch solves this by enabling **Sketch → Code transformation**.

---

## ✨ Key Features

- 🖼️ **Diagram Understanding**  
  Detects ML components (Conv, Dense, Pooling, etc.) from sketches

- 🔤 **Text Extraction (OCR)**  
  Reads labels like "ReLU", "Optimizer", etc.

- 🔗 **Graph Construction (DAG)**  
  Converts diagrams into structured pipelines

- 🧠 **Multimodal Fusion**  
  Combines visual + textual inputs

- ⚙️ **Code Generation**  
  Produces executable **PyTorch / Keras code**

- ⚠️ **Error Handling**  
  Handles ambiguous inputs via user clarification

---

## 🏗️ System Architecture

```text
Input (Image + Text)
        ↓
Vision Detection (YOLO / DETR)
        ↓
OCR Processing
        ↓
Graph Construction (DAG)
        ↓
Semantic Interpretation (IR)
        ↓
Code Generation (LLM)
        ↓
Output (PyTorch / Keras Code)
