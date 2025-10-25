# 🧠 Deep Neural Network from Scratch – Binary Classification

This project implements a **deep neural network (DNN)** using only **NumPy**, without relying on high-level libraries like TensorFlow or PyTorch. It’s designed to classify data into binary classes and showcases the mathematical and programming foundations of deep learning.

---

## 📌 Objective

To build a fully functional feedforward neural network for binary classification, covering:

- Forward propagation
- Activation functions
- Backpropagation
- Weight updates using gradient descent
- Evaluation on real or synthetic data

---

## 🧰 Technologies Used

- **Python**
- **NumPy** – matrix operations and numerical computations
- **Matplotlib** – optional (for visualizing performance)
- **Jupyter Notebook / .py script** – development and testing

---

## 🧠 Network Architecture

- **Input Layer**: Accepts `n` features
- **Hidden Layers**: Configurable (e.g., 2 hidden layers with ReLU)
- **Output Layer**: 1 neuron with Sigmoid activation
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Vanilla Gradient Descent

---

## 🧪 Features Implemented

- ✅ Vectorized forward and backward propagation
- ✅ Sigmoid and ReLU activation functions
- ✅ Dynamic number of hidden layers
- ✅ Manual weight and bias initialization
- ✅ Model training and accuracy calculation

---

## 🗂️ Project Structure

```bash
Deep-Neural-Network-Checkpoint/
├── deep_nn.py               # Core DNN logic (forward/backward propagation)
├── model_train.ipynb        # Notebook for testing and visualization
├── dataset.csv              # (Optional) Input data file
└── README.md                # Project documentation
▶️ How to Run
Clone the repo:

bash
Copy
Edit
git clone https://github.com/1Chizey/Deep-Neural-Network-Checkpoint.git
cd Deep-Neural-Network-Checkpoint
Install dependencies:

bash
Copy
Edit
pip install numpy matplotlib
Run the notebook or script:

bash
Copy
Edit
jupyter notebook model_train.ipynb
# or
python deep_nn.py
📊 Sample Output
Loss Plot: Decreasing loss over epochs

Accuracy Score: Final model accuracy on test data

Weight matrices: Shape and values of learned parameters

✅ Future Improvements
Add support for mini-batch gradient descent

Implement more activation functions (Leaky ReLU, Tanh)

Add multiclass classification with Softmax output

Integrate with real-world datasets (e.g., Iris, MNIST)

⭐ Acknowledgements
Inspired by Andrew Ng's Deep Learning Specialization

Built entirely with NumPy to understand the math behind neural networks

📬 Contact
Francis Chizey
Aspiring Machine Learning Engineer | Deep Learning Enthusiast
https://github.com/1Chizey | www.linkedin.com/in/francis-chizey-8838a5256 | chizeyfrancis@gmail.com
