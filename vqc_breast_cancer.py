import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(42)

# ==============================================================================
# Data Loading & Preprocessing
# ==============================================================================
# Load Breast Cancer dataset (Malignant: 0, Benign: 1)
data = load_breast_cancer()
X, y = data.data, data.target

# Train/Test Split (80% train, 20% test, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Extract the first 2 principal features using PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Normalize features to [-pi, pi]
scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
X_train_scaled = scaler.fit_transform(X_train_pca)
X_test_scaled = scaler.transform(X_test_pca)

# Cast to PennyLane NumPy arrays (marking input features as non-trainable)
X_train = np.array(X_train_scaled, requires_grad=False)
y_train = np.array(y_train, requires_grad=False)
X_test = np.array(X_test_scaled, requires_grad=False)
y_test = np.array(y_test, requires_grad=False)

# ==============================================================================
# Variational Quantum Circuit (VQC)
# ==============================================================================
num_qubits = 2
num_layers = 2

# Noiseless 2-qubit statevector simulator
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev)
def circuit(weights, x):
    """
    2-qubit VQC:
    - Angle embedding for 2 input features
    - 2 parameterized layers with RY and RZ rotations + CNOT entangler
    - Pauli-Z expectation value on qubit 0 as class logit
    """
    # 1. Feature encoding via Angle Embedding
    qml.AngleEmbedding(features=x, wires=range(num_qubits), rotation="X")

    # 2. Parameterized Variational Layers
    for l in range(num_layers):
        for q in range(num_qubits):
            qml.RY(weights[l, q, 0], wires=q)
            qml.RZ(weights[l, q, 1], wires=q)
        # Entangling gate
        qml.CNOT(wires=[0, 1])

    # 3. Measurement: Pauli-Z expectation value on qubit 0
    return qml.expval(qml.PauliZ(0))

# ==============================================================================
# Loss Function (Binary Cross-Entropy)
# ==============================================================================
def bce_loss(weights, X_batch, y_batch):
    # Expectation value in [-1, 1] acts as the class logit
    logits = circuit(weights, X_batch)
    
    # Sigmoid function converts logits to probability p(y=1)
    probs = 1.0 / (1.0 + np.exp(-logits))
    
    # Binary Cross-Entropy Loss
    return -np.mean(y_batch * np.log(probs) + (1.0 - y_batch) * np.log(1.0 - probs))

# ==============================================================================
# Training with Adam Optimizer
# ==============================================================================
# Parameter shape: (num_layers, num_qubits, 2 rotations [RY, RZ])
weights = np.random.uniform(
    low=-np.pi, high=np.pi, size=(num_layers, num_qubits, 2), requires_grad=True
)

# Compute initial loss
initial_loss = float(bce_loss(weights, X_train, y_train))
print(f"Initial Loss: {initial_loss:.4f}\n")

# Hyperparameters
epochs = 20
batch_size = 32
learning_rate = 0.1
opt = qml.AdamOptimizer(stepsize=learning_rate)

print(f"Starting training for {epochs} epochs (Batch Size: {batch_size}, LR: {learning_rate})...")
for epoch in range(1, epochs + 1):
    # Mini-batch shuffling
    shuffle_indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[shuffle_indices]
    y_shuffled = y_train[shuffle_indices]

    for i in range(0, len(X_train), batch_size):
        X_batch = X_shuffled[i : i + batch_size]
        y_batch = y_shuffled[i : i + batch_size]
        weights, _ = opt.step_and_cost(lambda w: bce_loss(w, X_batch, y_batch), weights)

    epoch_loss = float(bce_loss(weights, X_train, y_train))
    print(f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {epoch_loss:.4f}")

final_train_loss = float(bce_loss(weights, X_train, y_train))

# ==============================================================================
# Model Evaluation on Test Set
# ==============================================================================
test_logits = circuit(weights, X_test)
test_probs = 1.0 / (1.0 + np.exp(-test_logits))
test_preds = np.where(test_probs >= 0.5, 1, 0)
test_acc = accuracy_score(y_test, test_preds)

print("\n" + "=" * 40)
print(f"Initial Loss:        {initial_loss:.4f}")
print(f"Final Training Loss: {final_train_loss:.4f}")
print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
print("=" * 40)
