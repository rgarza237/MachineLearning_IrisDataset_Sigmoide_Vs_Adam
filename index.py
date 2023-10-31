import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def create_and_train_model(optimizer, X_train, y_train, X_test, y_test):
    model = Sequential([
        Dense(10, input_dim=4, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, verbose=0)
    return history, model

sgd_history, sgd_model = create_and_train_model(SGD(), X_train, y_train, X_test, y_test)
adam_history, adam_model = create_and_train_model(Adam(), X_train, y_train, X_test, y_test)

#tabla comparativa
plt.plot(sgd_history.history['val_accuracy'], label='SGD')
plt.plot(adam_history.history['val_accuracy'], label='Adam')
plt.title('Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

sgd_score = sgd_model.evaluate(X_test, y_test, verbose=0)
adam_score = adam_model.evaluate(X_test, y_test, verbose=0)

print(f"SGD Model - Loss: {sgd_score[0]:.4f}, Accuracy: {sgd_score[1]:.4f}")
print(f"Adam Model - Loss: {adam_score[0]:.4f}, Accuracy: {adam_score[1]:.4f}")

print("Conclusiones:")
print("El optimizador Adam generalmente converge más rápidamente y") 
print("proporciona una mayor precisión que el SGD para el conjunto de datos Iris.")
print("Este comportamiento puede atribuirse a la forma en que Adam ajusta dinámicamente la tasa de aprendizaje durante el entrenamiento,")
print("lo cual es especialmente útil para conjuntos de datos complejos y multifacéticos como Iris.")


