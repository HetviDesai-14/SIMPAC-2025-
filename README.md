AI-Driven CNN-KNN Fusion for Plant Disease Classification

Table of Contents
•	Introduction
•	Key Features
•	Installation
•	Usage
•	Dataset
•	Model Architecture
•	Results
•	Future Improvements
•	Contributing
•	License
•	Acknowledgments
________________________________________
Introduction
Plant diseases like Anthracnose and Powdery Mildew cause significant crop losses, with manual detection methods being time-consuming and error-prone. Existing automated solutions suffer from poor training efficiency, slow detection rates, and difficulty identifying subtle symptoms. This project introduces ACKFS (AI-Driven CNN-KNN Fusion Software), a hybrid model combining Convolutional Neural Networks (CNN) and K-Nearest Neighbors (KNN) to enhance early disease detection. By integrating optimized preprocessing, hierarchical feature extraction, and hybrid classification, ACKFS achieves state-of-the-art accuracy while addressing challenges like environmental variability and data scarcity.
________________________________________
Key Features
•	Hybrid CNN-KNN Architecture: Combines CNN’s feature extraction with KNN’s classification robustness.
•	Data Augmentation: Implements rotation, scaling, and brightness adjustments to enhance dataset diversity.
•	Optimized Preprocessing: Uses HSV transformation and GLCM/LBP for noise reduction and feature clarity.
•	Real-World Applicability: Achieves 94.56% accuracy on mango leaves and 87.52% on multi-species datasets.
•	Scalability: Designed for deployment on lightweight devices to enable field use.
________________________________________
Installation
Installation To use SSTAS, follow these steps:
1.	Clone the repository: git clone https://github.com/HetviDesai-14/SIMPAC-2025-.git
2.	Install dependencies: Ensure Python 3.8+ is installed. Then, install the required libraries: pip install -r requirements.txt
3.	Download the dataset: The dataset is available on Mendeley Data(https://data.mendeley.com/datasets/hp2cdckpdr/2). Place it in the data/ directory. Usage
4.	Data Preprocessing Preprocess the spectral data using the provided script: python scripts/preprocess_data.py
5.	Training the Model Train the deep learning model: python scripts/train_model.py
6.	Testing the Model Evaluate the model on the test dataset: python scripts/test_model.py
7.	Visualization Visualize the results: python scripts/visualize_results.py
   
Dataset
Two datasets are used:
1.	Mango Leaf BD:
o	8 classes (7 diseases + healthy).
o	6,400 training and 1,600 testing images.
2.	Leaf Repository:
o	12 classes (multi-species).
o	4,000 training and 1,000 testing images.
Dataset	Classes	Training Images	Testing Images
Mango Leaf BD	8	6,400	1,600
Leaf Repository	12	4,000	1,000
________________________________________
Model Architecture

Workflow
1.	Data Preprocessing:
o	HSV color space conversion and thresholding-based segmentation.
o	Augmentation (rotation, flipping, brightness adjustment).
2.	Feature Extraction:
o	Three CNN blocks (Conv2D + MaxPooling) reduce images to 7x7 feature maps.
o	GLCM and LBP for texture analysis.
3.	Hybrid Classification:
o	Fully Connected (256-node) layer for preliminary classification.
o	KNN refinement using Euclidean distance and majority voting.
 
________________________________________
Results
•	Accuracy:
o	Mango Leaf BD: 94.56% (testing), outperforming MobileNet R-CNN (70.53%).
o	Leaf Repository: 87.52% (testing).
•	Loss: Training (0.1112), Validation (0.1023), Testing (0.1059).
•	Precision/Recall: >93% for both metrics.
________________________________________
Future Improvements
•	Real-Time Deployment: Optimize for mobile/edge devices.
•	Synthetic Data: Integrate GANs for rare disease cases.
•	Multi-Modal Fusion: Combine images with spectral/soil data.
•	Explainability: Add Grad-CAM/SHAP for interpretability.

