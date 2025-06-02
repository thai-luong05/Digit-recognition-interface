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

'''In order for the digit to match the centered data in mnist dataset we train on, when processing the digit from the canvas, the image processed
will be center for better recognition'''
def preprocess_image(raw_data):
    try:
        # Just in case it's a base64 string with header info
        if raw_data.startswith('data:image/png;base64,'):
            raw_data = raw_data.split(',')[1]  # strip off the header part

        # Decode the image from base64
        decoded_bytes = base64.b64decode(raw_data)
        img = Image.open(io.BytesIO(decoded_bytes))

        # Convert to grayscale (one channel)
        gray_img = img.convert('L')

        # Ensure image is exactly 28x28
        if gray_img.size != (28, 28):
            print(f"Resizing from {gray_img.size} to 28x28")
            gray_img = gray_img.resize((28, 28), Image.Resampling.LANCZOS)  # LANCZOS looks nicer

        # Turn image into a NumPy array
        pixel_array = np.array(gray_img, dtype=float)

        print("Initial stats:")
        print(f" - Shape: {pixel_array.shape}")
        print(f" - Range: {pixel_array.min():.1f} to {pixel_array.max():.1f}")

        # Normalize pixel values to range [0, 1]
        pixel_array = pixel_array / 255.0

        print("After normalization:")
        print(f" - Min: {pixel_array.min():.3f}, Max: {pixel_array.max():.3f}, Mean: {pixel_array.mean():.3f}")

        # Try to center the digit (in case it's not aligned)
        aligned = center_digit(pixel_array)

        if aligned is None:
            print("Oops — centering failed. Using the original instead.")
            aligned = pixel_array

        # Check result size before reshaping
        if aligned.shape != (28, 28):
            print(f"Warning: Unexpected shape after centering: {aligned.shape}")
            aligned = aligned[:28, :28]  # quick patch-up

        print("Final image stats:")
        print(f" - Shape: {aligned.shape}, Range: {aligned.min():.3f} to {aligned.max():.3f}")

        # Flatten for model input
        flattened = aligned.reshape(1, 784)

        return flattened

    except Exception as err:
        print("Something went wrong in preprocessing...")
        import traceback
        traceback.print_exc()
        raise err


def center_digit(img_arr, threshold=0.1):
    # Re-centers the digit by shifting based on center of mass
    try:
        print("Centering digit...")

        # Create a mask where pixel value > threshold (i.e., likely digit)
        mask = img_arr > threshold

        if not np.any(mask):
            print("No strong pixels found. Bail out.")
            return img_arr

        # Locate all non-zero pixel positions
        ys, xs = np.where(mask)

        if len(ys) == 0 or len(xs) == 0:
            return img_arr  # Shouldn't happen but better safe than sorry

        # Use weighted center of mass to get accurate shift
        weights = img_arr[mask]
        avg_y = np.average(ys, weights=weights)
        avg_x = np.average(xs, weights=weights)

        print(f"Digit center at ({avg_x:.2f}, {avg_y:.2f})")

        # Figure out how far off-center we are
        target = 13.5  # halfway point of a 28x28 image
        delta_y = target - avg_y
        delta_x = target - avg_x

        # Skip if we're already close enough
        if abs(delta_x) < 0.5 and abs(delta_y) < 0.5:
            print("Already centered enough.")
            return img_arr

        shifted = shift_image(img_arr, delta_y, delta_x)
        return shifted if shifted is not None else img_arr

    except Exception as centering_err:
        print("Centering error occurred.")
        import traceback
        traceback.print_exc()
        return img_arr


def shift_image(image, offset_y, offset_x):
    """
    Shifts the image around using simple numpy slicing.
    """
    try:
        height, width = image.shape
        moved = np.zeros_like(image)

        # Use int shift only
        dy = int(round(offset_y))
        dx = int(round(offset_x))

        print(f"Shifting image by dx={dx}, dy={dy}")

        # Figure out valid slice bounds for source and destination
        y_src_start = max(0, -dy)
        y_src_end = min(height, height - dy)
        x_src_start = max(0, -dx)
        x_src_end = min(width, width - dx)

        y_dst_start = max(0, dy)
        y_dst_end = y_dst_start + (y_src_end - y_src_start)
        x_dst_start = max(0, dx)
        x_dst_end = x_dst_start + (x_src_end - x_src_start)

        # Sanity check on slice ranges
        if (y_src_end > y_src_start and x_src_end > x_src_start and
                y_dst_end > y_dst_start and x_dst_end > x_dst_start):
            moved[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = \
                image[y_src_start:y_src_end, x_src_start:x_src_end]
        else:
            print("Shift too large or out of bounds — skipping.")
            return image

        return moved

    except Exception as shift_err:
        print("Error while shifting the image:")
        import traceback
        traceback.print_exc()
        return image

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
