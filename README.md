This repository is produced to share material relevant to the Journal paper **"Automatic crack detection on concrete and asphalt surfaces using semantic segmentation network with hierarchical Transformer"** published in **Engineering structures**.

The functions implemented in this repository include training of pixel-level crack detection models, visualization of model predictions, evaluation of the model, and assessment of various metrics for detection results.

**Model Training:** The "train.py" script in the "tools" directory is utilzed for training crack detection models. The training outcomes, including trained weights, training process, and achieved metrics, are stored in the "logs" directory.
![image](https://github.com/Li-Hubing/CrackSegFormer/assets/103866679/46fd51df-b294-4edd-ad9d-02616d00b0d1)


**Model Evaluation:** Model parameters, FLOPs, Latency, and FPS can be calculated using "model_evaluation.py".

![image](https://github.com/Li-Hubing/CrackSegFormer/assets/103866679/170b1a24-14a5-4999-a70b-5a8f807a91d3)

**Model Prediction:** The visualization results of crack detection using a trained model can be achieved using "predict.py".

![image](https://github.com/Li-Hubing/CrackSegFormer/assets/103866679/73ee0a88-04c7-495d-b4e4-0b50e0336d64)

**Performance Evaluation:** The calculation of Dice coefficient, F1 score, Precision, Recall, Accuracy, and mIoU metrics can be implemented using "validation.py".

(⚡Due to time constraints, the code and instructions will be gradually refined.⚡⚡)
