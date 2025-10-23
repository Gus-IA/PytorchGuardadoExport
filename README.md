# 🧠 CNN para clasificación de dígitos MNIST con PyTorch, TorchScript y ONNX

Este proyecto demuestra el flujo completo de entrenamiento, guardado y exportación de un modelo de red neuronal convolucional (CNN) utilizando **PyTorch 2.6**, **TorchScript** y **ONNX**.

---

## 🚀 Características principales

- Entrenamiento de una **CNN** sobre el dataset **MNIST**.
- Evaluación de rendimiento en entrenamiento y test.
- Guardado y carga de modelos con:
  - `state_dict` (solo pesos)
  - modelo completo (`torch.save(model)`)
- Exportación del modelo a:
  - **TorchScript (trace y script)**
  - **ONNX** para interoperabilidad con otros frameworks.
- Inferencia con **onnxruntime**.
- Ejemplo de **preprocesamiento y postprocesamiento** integrados en el modelo final.

---

## 🧰 Requisitos

Instala las dependencias necesarias:

```bash
pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
