Helmet Violation Model Prediction
Helmet Violation Model Prediction is an open-source project that utilizes deep learning models to predict helmet violations based on vehicle data. The project implements several machine learning models and provides easy-to-use APIs for users to experiment with helmet violation detection.

Table of Contents
       Who Should Use Helmet Violation Model Prediction
       Why Use Helmet Violation Model Prediction
       Project Capabilities
       Requirements and Installation
					       Training
					       Usage
					       Citation
					       License
       Who Should Use Helmet Violation Model Prediction
       Beginners: Those who want to get started with deep learning for real-time object detection tasks like helmet violation detection.
       Engineers: Engineers interested in trying out different helmet violation models to see how deep learning can be applied to improve traffic safety.
       Researchers: Researchers looking for an open-source solution to easily experiment with different helmet detection algorithms and models.
       Why Use Helmet Violation Model Prediction
       Lightweight: Easy to integrate and use without heavy dependencies.
       Open Source: Free to use, modify, and distribute under the MIT license.
       Active Maintenance: The project is regularly updated with improvements, new features, and bug fixes.
       Simple Setup: Get started with pre-trained models or use your own datasets for training.
Project Capabilities
At this moment, this project includes the following models and techniques for helmet violation detection:

Type	ABBRV	Algorithms	Description
Object Detection	YOLO	YOLOv3 Helmet Detection	Pre-trained YOLO model for helmet detection.
Image Classification	CNN	CNN-based Helmet Detection	A basic CNN model for classifying helmet violations.
Transfer Learning	ResNet	Transfer Learning using ResNet-50 for Helmet Detection	Pre-trained ResNet-50 fine-tuned for helmet detection.
Hyperparameter Optimization	HPO-CG	Hyperparameter optimization for tuning deep learning models	Optimizing model performance via HPO.
For more details on each algorithm, please refer to the specific algorithm documentation provided in the repository.

Requirements and Installation
System Requirements:
 Python >= 3.6
 PyTorch >= 1.5.0
 OpenCV (for image visualization and processing)
Additional libraries: numpy, matplotlib, scikit-learn
Installation Steps:
Clone the repository:

    
    git clone https://github.com/Hackb07/helmet_violation_model_prediction.git
Navigate to the project directory:

     
     cd helmet_violation_model_prediction
Install dependencies:

    
    pip install -r requirements.txt
(Optional) If you want to use pre-trained models, download them and place them in the .latent-data/ directory.

Dataset:
The project supports image and video datasets for helmet violation detection. Place your dataset in the data/ directory with subfolders for "helmet" and "no_helmet" classes, as shown below:

    
    /data
        /train
            /helmet
            /no_helmet
        /test
            /helmet
            /no_helmet
Training
To train the model, ensure that your dataset is properly structured as mentioned above. Then, run the following command to start training:

    
    python train_model.py --data <path_to_data_directory>
You can specify the dataset directory using the --data argument.

Usage
To make predictions on images or videos, use the following commands:

For Image-Based Prediction:
     
     
    python predict_helmet_violation.py --image <path_to_image>
For Video-Based Prediction:
        
      
     python predict_helmet_violation.py --video <path_to_video>
Command Line Arguments:
--image: Path to the image file for prediction.
--video: Path to the video file for prediction.
Citation
If you find this project useful in your research or development, please consider citing the related papers:

bibtex
Copy code
@inproceedingstharun2024helmet,
  title     = Helmet Violation Detection using Deep Learning
  author    = Tharun Bala 
  booktitle = Proceedings of the International Conference on AI and Machine Learning
  year      = 2024

License
The entire codebase is under the MIT license. See the LICENSE file for more details.
