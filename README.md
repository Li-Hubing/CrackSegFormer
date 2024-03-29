# Crack segmentation based on deep learning

This repository is produced to share material relevant to the Journal paper [Automatic crack detection on concrete and asphalt surfaces using semantic segmentation network with hierarchical Transformer](https://www.sciencedirect.com/science/article/pii/S0141029624004656)
 published in **Engineering structures**.  

The functions implemented in this repository include evaluation of the model, training of pixel-level crack detection models, visualization of model predictions, and assessment of various metrics for detection results.   

## Getting Started


**1. Model Evaluation:** Model parameters, FLOPs, Latency, and FPS can be calculated using：  

```
python tools/model_evaluation.py
```

**2. Model Training:** Training of pixel-level crack detection models can be initiated with:  

```
python tools/train.py
```

The training outcomes, including trained weights, training process, and achieved metrics, are stored in the "logs" directory.

![image](https://github.com/Li-Hubing/CrackSegFormer/assets/103866679/46fd51df-b294-4edd-ad9d-02616d00b0d1)


![image](https://github.com/Li-Hubing/CrackSegFormer/assets/103866679/170b1a24-14a5-4999-a70b-5a8f807a91d3)

**3. Model Prediction:** Visualizing crack detection results with a trained model can be accomplished by executing:

```
python tools/batch_predict.py
```

![image](https://github.com/Li-Hubing/CrackSegFormer/assets/103866679/73ee0a88-04c7-495d-b4e4-0b50e0336d64)

**4. Performance Evaluation:** The calculation of Dice coefficient, F1 score, Precision, Recall, Accuracy, and mIoU metrics can be implemented using:

```
python tools/validation.py
```

(⚡Due to time constraints, the code and instructions will be gradually refined.⚡⚡)
