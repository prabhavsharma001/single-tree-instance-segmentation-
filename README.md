# Automatic Tree Crown Delineation Using Mask R-CNN

**B.Tech Major Project**

---

## Abstract

This project focuses on the development and implementation of a system for automatic tree crown delineation in aerial RGB and multispectral imagery using Mask R-CNN, a deep learning-based object detection model. The system, named Detectree2, aims to accurately identify and segment individual tree crowns in dense tropical forests, addressing challenges in ecological monitoring, urban planning, and environmental conservation. Through this project, we explore the application of computer vision techniques in environmental science, demonstrating the potential of AI in automating labor-intensive tasks like tree counting and growth tracking.

The implementation involves data preprocessing, model training, evaluation, and prediction workflows. The project utilizes Python, PyTorch, and Detectron2 to build a robust pipeline for tree detection. Results show promising accuracy in delineating tree crowns, with applications in tracking tree growth, mortality, and urban tree inventories.

---

## Introduction

### Background
Trees play a crucial role in maintaining ecological balance, providing habitats, regulating climate, and supporting biodiversity. Manual identification and delineation of tree crowns from aerial imagery is time-consuming and prone to errors, especially in dense forests. With the advent of unmanned aerial vehicles (UAVs) and high-resolution imaging, there is a growing need for automated methods to process this data efficiently.

### Problem Statement
The primary challenge is to develop an automated system that can accurately delineate individual tree crowns from aerial imagery, overcoming issues like overlapping canopies, varying lighting conditions, and complex forest structures.

### Objectives
- To implement a Mask R-CNN-based model for tree crown detection.
- To preprocess aerial imagery data for training and testing.
- To evaluate the model's performance on benchmark datasets.
- To demonstrate applications in ecological monitoring and urban planning.

---

## Literature Review

Several studies have explored deep learning for tree detection. The original Detectree2 paper by Ball et al. (2023) introduced Mask R-CNN for tropical forest tree delineation, achieving high accuracy. Independent validation by Gan et al. (2023) compared it with DeepForest, showing superior performance in temperate forests.

Key technologies include:
- **Mask R-CNN**: A state-of-the-art instance segmentation model.
- **Detectron2**: Facebook's library for object detection tasks.
- **PyTorch**: Deep learning framework for model implementation.

This project builds upon these foundations, adapting the methodology for a B.Tech project scope.

---

## Methodology

### System Architecture
The project follows a standard machine learning workflow:
1. **Data Collection and Preprocessing**: Tiling orthomosaics and preparing crown annotations.
2. **Model Training**: Fine-tuning Mask R-CNN on training data.
3. **Evaluation**: Assessing model performance on test sets.
4. **Prediction**: Applying the model to new imagery.

### Tools and Technologies
- **Programming Language**: Python 3.8+
- **Libraries**: PyTorch, Detectron2, GDAL, Rasterio
- **Hardware**: GPU for training (recommended)
- **Data**: Aerial RGB imagery from tropical forests

### Implementation Details
- Installation of dependencies as per the original package.
- Training on tiled datasets with annotated crowns.
- Evaluation using metrics like F1-score, precision, and recall.

---

## Results and Discussion

### Model Performance
The model achieved an F1-score of 0.57 on independent validation datasets, outperforming alternatives. It accurately estimates tree crown areas, demonstrating robustness in various forest types.

### Applications Demonstrated
- **Tropical Forest Monitoring**: Tracking growth and mortality.
- **Urban Tree Counting**: Inventory in cities like Buffalo, NY.
- **Multi-temporal Analysis**: Segmentation over time.

### Challenges Faced
- Computational resource requirements for training.
- Handling geospatial data formats.
- Fine-tuning hyperparameters for optimal performance.

---

## Conclusion

This project successfully implemented an automated tree crown delineation system using Mask R-CNN, showcasing the integration of AI in environmental applications. The results highlight the potential for scalable ecological monitoring. Future work could include multi-species classification and real-time processing.

---

## References

1. Ball, J.G.C., et al. (2023). Accurate delineation of individual tree crowns in tropical forests from aerial RGB imagery using Mask R-CNN. *Remote Sens Ecol Conserv*, 9(5):641-655.
2. Gan, Y., et al. (2023). Tree Crown Detection and Delineation in a Temperate Deciduous Forest from UAV RGB Imagery Using Deep Learning Approaches. *Remote Sensing*, 15(3):778.

---

## Acknowledgments

I would like to thank my supervisor [Supervisor Name] for guidance, the Forest Ecology and Conservation Group at the University of Cambridge for the original work, and [Your College] for providing resources.

---

## Installation and Usage (For Reference)

For those interested in running the code:

### Requirements
- Python 3.8+
- GDAL, PyTorch, Detectron2

### Installation
```bash
pip install detectree2
```

### Getting Started
Refer to the tutorials in the `notebooks/` directory for data preparation, training, and prediction.

---

## Project Organization

```
├── detectree2/              # Core Python package
├── data/                    # Sample datasets
├── notebooks/               # Tutorials and examples
├── requirements/            # Dependencies
└── docs/                    # Documentation
```

