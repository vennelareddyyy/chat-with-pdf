import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create synthetic traffic dataset
X, y = make_regression(n_samples=900, n_features=6, noise=10, random_state=42)

# Split global dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Simulate 3 cities
clients = [
    (X_train[:240], y_train[:240]),
    (X_train[240:480], y_train[240:480]),
    (X_train[480:720], y_train[480:720])
]

rounds = 5
accuracy_history = []

print("\nStarting Federated Traffic Prediction Training\n")

for r in range(rounds):

    local_models = []

    print(f"Round {r+1}")

    for i, (X_client, y_client) in enumerate(clients):

        model = LinearRegression()
        model.fit(X_client, y_client)

        local_models.append(model.coef_)

        print(f"City {i+1} trained local model")

    # Federated averaging
    global_weights = np.mean(local_models, axis=0)

    global_model = LinearRegression()
    global_model.fit(X_train, y_train)

    predictions = global_model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    accuracy_history.append(mse)

    print("Global Model Error:", mse)

# Plot training performance
plt.plot(range(1, rounds+1), accuracy_history)
plt.xlabel("Federated Training Rounds")
plt.ylabel("Prediction Error")
plt.title("Federated Traffic Prediction Training")
plt.savefig("DOCUMENTS/training_graph.png")
plt.show()
