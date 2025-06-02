from flask import Flask, request, jsonify
from flask_cors import CORS
import math, random, numpy as np, matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds
import io
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class ANN():
    def __init__(self):
        n_in = 784 #28x 28
        n_out = 10
        #xavier, glorot initialization
        self.weight1 = np.random.uniform(-np.sqrt(6/(n_in + n_out)), np.sqrt(6/(n_in + n_out)), size=(n_in, n_out)) #size should be 10 vector
        self.weight2 = np.random.uniform(-np.sqrt(6/(n_in + n_out)), np.sqrt(6/(n_in + n_out)), size=(10, 10)) #for the second layer hidden
        self.weight3 = np.random.uniform(-np.sqrt(6/(n_in + n_out)), np.sqrt(6/(n_in + n_out)), size=(10, 10)) #size should be 10 vector

        self.bias1 = np.array([0] * n_out, dtype=float).reshape(1, 10)
        self.bias2 = np.array([0] * n_out, dtype=float).reshape(1, 10)
        self.bias3 = np.array([0] * n_out, dtype=float).reshape(1, 10)
        
    def sigmoid(self, x):
        return 1/(1+math.e**(-x))
    
    def error(self, output, desire):
        total = 0
        for i in range(10):
            total += (output[i] - desire[i])**2
        return total / 2

    def train(self, data, entries, epochs): #train perhaps?
        rate = 0.01
        epoch_losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for i in entries:
                input = data[i][0]
                desired_output =data[i][1]

                z1 = input @ self.weight1 + self.bias1
                out1 = self.sigmoid(z1)
                z2 = out1 @ self.weight2 + self.bias2
                out2 = self.sigmoid(z2) 
                z3 = out2 @ self.weight3 + self.bias3
                out3 = self.sigmoid(z3)
                loss = np.mean((out3 - desired_output)**2)
                epoch_loss += loss #loss for each data

                de_do3 = out3 - desired_output
                do3_dz3 = out3 * (1 - out3)
                delta3 = de_do3 * do3_dz3 #de_dz3
                dz3_dw3 = out2.T

                de_dw3 = dz3_dw3 @ delta3
                de_db3 = np.sum(delta3, axis = 0, keepdims= True)
                
                #calculating weights using chain rule
                dz3_do2 = self.weight3.T
                do2_dz2 = out2 * (1 - out2)
                #delta: error per neuron
                delta2 = (delta3 @ dz3_do2) * do2_dz2 #de_dz2
                dz2_dw2 = out1.T

                de_dw2 = dz2_dw2 @ delta2
                #sum all the gradients in order for the last data not to be dominant, also sum across batch to adapt to the overall data
                de_db2 = np.sum(delta2, axis = 0, keepdims = True)

                delta1 = (delta2 @ self.weight2.T) * out1 * (1 - out1) #de_dz1
                de_dw1 = input.T @ delta1
                de_db1 = np.sum(delta1, axis = 0, keepdims = True)

                #update weights
                self.weight1 -= rate * de_dw1
                self.weight2 -= rate * de_dw2
                self.weight3 -= rate * de_dw3
                self.bias1 -= rate * de_db1
                self.bias2 -= rate * de_db2
                self.bias3 -= rate * de_db3
            epoch_losses.append(epoch_loss / len(data))
            correct =0
            for i in range(y_test.shape[0]):
                rgb = x_test[i].reshape(1, 784)
                value = y_test[i]
                prediction = network.retest(rgb)
                if prediction == value:
                    correct += 1
            print(f"Epoch {epoch+1}, Avg Loss: {epoch_loss / len(data):.5f}, Current accuracy: {correct / y_test.shape[0]}") #avg loss across all data
        return epoch_losses
    
    def retest(self, input):
        z1 = input @ self.weight1 + self.bias1
        out1 = self.sigmoid(z1)
        z2 = out1 @ self.weight2 + self.bias2
        out2 = self.sigmoid(z2)
        z3 = out2 @ self.weight3 + self.bias3
        out3 = self.sigmoid(z3)
        
        predicted_class = np.argmax(out3)
        return predicted_class      

#train multiple times to see accuracy
accuracy = []
network = ANN()
data = {}
size = -1
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
train_ds, test_ds = tfds.load(
    'emnist/digits',           # Dataset name (emnist/digits, emnist/letters, etc.)
    split=['train', 'test'],   # Load both train and test splits
    as_supervised=True,        # Returns (image, label) tuples
    shuffle_files=True,       # Shuffle the data
)
x_t, y_t = [], []
x_te, y_te = [], []
for image, label in train_ds:
    x_t.append(image.numpy())
    y_t.append(label.numpy())

for image, label in test_ds:
    x_te.append(image.numpy())
    y_te.append(label.numpy())

x_t = np.array([i.T for i in np.squeeze(np.array(x_t))])
x_te = np.array([i.T for i in np.squeeze(np.array(x_te))])
y_te = np.squeeze(np.array(y_te))
y_t = np.array(y_t)

x_train = np.concatenate((x_train, x_t), axis =0)
x_test = np.concatenate((x_test, x_te), axis =0)
y_train = np.concatenate((y_train, y_t), axis =0)
y_test = np.concatenate((y_test, y_te), axis =0)
#to gray scale
x_train = x_train / 255.0
x_test = x_test / 255.0

for i in range(y_train.shape[0]):
    rgb = x_train[i].reshape(1, 784)
    desired_output = [0] * 10
    desired_output[y_train[i]] = 1
    desired_output = np.array(desired_output).reshape(1, 10)
    size += 1
    data[size] = (rgb, desired_output)
entries = [i for i in data.keys()]
#hsuffle before train
random.shuffle(entries)
print("Training...")
avgloss = network.train(data, entries, epochs=150)
print("Completed!")

def preprocess_image(image_data):
    # Remove data URL prefix if present
    if 'data:image/png;base64,' in image_data:
        image_data = image_data.split(',')[1]
    
    # Decode base64 image
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Ensure it's 28x28
    if image.size != (28, 28):
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    image_array = np.array(image, dtype=float)
    
    # Debug: Print some statistics
    print(f"Image shape: {image_array.shape}")
    print(f"Pixel value range: {image_array.min():.1f} to {image_array.max():.1f}")
    print(f"Mean pixel value: {image_array.mean():.1f}")
    
    # Canvas format: white digit (255) on black background (0)
    # MNIST format after normalization: digit=1, background=0
    # So we normalize directly: white becomes 1, black becomes 0
    image_array = image_array / 255.0  # Direct normalization: black=0, white=1
    
    print(f"After normalization: {image_array.min():.3f} to {image_array.max():.3f}")
    print(f"Mean after normalization: {image_array.mean():.3f}")
    
    # Reshape for network input
    image_array = image_array.reshape(1, 784)
    
    return image_array

@app.route('/predict', methods=['POST'])
def predict_digit():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Process the image
        processed_image = preprocess_image(data['image'])
        
        # Make prediction
        prediction = network.retest(processed_image)
        
        print(f"Prediction: {prediction}")
        
        return jsonify({
            'prediction': int(prediction),
            'message': f'Predicted digit: {prediction}'
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test_server():
    """Test endpoint to verify server is running"""
    return jsonify({'message': 'Server is running', 'status': 'OK'})

if __name__ == '__main__':
    print("Server starting on http://localhost:5000")
    app.run(debug=False, port=5000)