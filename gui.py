import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('Bald_detector.h5')

# Create the Tkinter GUI window
window = tk.Tk()
window.title("Image Prediction")
window.geometry("400x400")


# Function to handle image selection and prediction
def predict_image():
    # Open file dialog for image selection
    file_path = filedialog.askopenfilename()
    
    # Load and preprocess the selected image
    img = Image.open(file_path)
    img = img.resize((256, 256))  # Resize the image to match the model's input shape
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    
    # Perform prediction
    result = model.predict(img_array)
    prediction = result[0][0]  # Get the predicted value
    
    # Convert prediction value to "Bald" or "Not Bald"
    if prediction < 0.55:
        prediction_text = "Bald"
    else:
        prediction_text = "Not Bald"
    
    # Display the prediction result
    result_label.config(text="Prediction: {}".format(prediction_text))

    # Display the selected image
    img = img.resize((300, 300))  # Resize the image for display
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

# Button for image selection
select_button = tk.Button(window, text="Select Image", command=predict_image)
select_button.pack(pady=10)

# Label to display the selected image
image_label = tk.Label(window)
image_label.pack()

# Label to display the prediction result
result_label = tk.Label(window, text="Prediction: ")
result_label.pack(pady=10)

# Run the Tkinter event loop
window.mainloop()
