# Task 05: Food Item Recognition and Calorie Estimation

## Objective

Develop a model that can accurately recognize food items from images and estimate their calorie content, enabling users to track their dietary intake and make informed food choices.

## Dataset

For this task, we have used a custom dataset consisting of images of two food items: burgers and pizzas. The dataset is structured as follows:

- **Train Images**:
  - **Burgers**: 10 images
  - **Pizzas**: 10 images

- **Test Images**:
  - **Burgers**: 5 images
  - **Pizzas**: 5 images

## File Structure

The project directory is organized as follows:

PRODIGY_ML_05/
│
├── train/
│ ├── burgers/
│ │ ├── burger1.jpg
│ │ ├── burger2.jpg
│ │ ├── ...
│ └── pizzas/
│ ├── pizza1.jpg
│ ├── pizza2.jpg
│ ├── ...
│
├── test/
│ ├── burgers/
│ │ ├── test_burger1.jpg
│ │ ├── test_burger2.jpg
│ │ ├── ...
│ └── pizzas/
│ ├── test_pizza1.jpg
│ ├── test_pizza2.jpg
│ ├── ...
│
├── food_classification.py
├── requirements.txt
└── README.md

## Requirements

The following libraries are required for this project:

numpy
scikit-learn
opencv-python
matplotlib

pip install -r requirements.txt

Run the Classification Script:

Execute the food_classification.py script to train the model and evaluate its performance:

python food_classification.py
View Results:

The script will output the accuracy of the model in the terminal. Additionally, it will display example predictions.
Code Explanation
Data Preparation:


An SVM classifier is used for image classification.
The model is trained on the training images and evaluated on the test images.
Evaluation:

The model's accuracy is calculated based on predictions on test images.
Example predictions are visualized using matplotlib.
Example Output
The script will provide output similar to:

Accuracy: 60.00%
This indicates the percentage of correctly classified test images.
