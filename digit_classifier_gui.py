import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Add this import
import time

# Load the saved model
model = tf.keras.models.load_model("mnist_model.h5")

# This will store the figure object to update it later
fig, ax = plt.subplots(figsize=(6, 4))

def show_distribution(predictions):
    ax.clear()  # Clear the current axes to prevent new figures from opening

    bars = ax.bar(range(10), predictions[0], color='skyblue')
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Confidence")
    ax.set_xticks(range(10))

    # Add percentage labels
    for bar, prob in zip(bars, predictions[0]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{prob * 100:.1f}%', 
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.draw()  # Redraw the updated plot

class DigitApp:
    def __init__(self, master):
        self.master = master
        self.last_prediction_time = time.time()
        self.master.title("Digit Classifier")

        # Frame to hold canvas and graph side by side
        self.frame = tk.Frame(master)
        self.frame.pack()

        # Canvas for drawing
        self.canvas_frame = tk.Frame(self.frame)
        self.canvas_frame.pack(side='left')

        self.canvas = tk.Canvas(self.canvas_frame, width=280, height=280, bg='white')
        self.canvas.pack()

        # Frame for buttons
        self.button_frame = tk.Frame(master)
        self.button_frame.pack()

        # Add the dropdown for correction
        self.correct_label_var = tk.StringVar(value="0")
        self.label_dropdown = tk.OptionMenu(self.button_frame, self.correct_label_var, *[str(i) for i in range(10)])
        self.label_dropdown.pack(side='left')

        # Add the "Learn from Drawing" button
        self.learn_btn = tk.Button(self.button_frame, text="Learn from Drawing", command=self.learn_from_drawing)
        self.learn_btn.pack(side='left')

        # Predict and Clear buttons
        self.predict_btn = tk.Button(self.button_frame, text="Show Prediction", command=self.predict_digit)
        self.predict_btn.pack(side='left')

        self.clear_btn = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_btn.pack(side='left')

        self.save_btn = tk.Button(self.button_frame, text="Save", command=self.save_image)
        self.save_btn.pack(side='left')

        self.result_label = tk.Label(master, text="Draw a digit!", font=("Helvetica", 16))
        self.result_label.pack()

        # PIL image for drawing
        self.image = Image.new("L", (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)

        # Initialize last_x, last_y
        self.last_x, self.last_y = None, None

        # Bind drawing events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)

        # Frame for the live graph
        self.graph_frame = tk.Frame(self.frame)
        self.graph_frame.pack(side='right')

        # Add the graph (live update) to the side frame
        self.canvas_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.graph = self.canvas_figure.add_subplot(111)

        self.canvas_widget = FigureCanvasTkAgg(self.canvas_figure, self.graph_frame)
        self.canvas_widget.get_tk_widget().pack()

    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw_on_canvas(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            # Interpolating points for smoother curve
            self.create_smooth_curve(self.last_x, self.last_y, x, y)

        self.last_x, self.last_y = x, y

        # Call prediction while drawing
        if time.time() - self.last_prediction_time > 0.3:
            self.predict_digit()
            self.last_prediction_time = time.time()

    def create_smooth_curve(self, x1, y1, x2, y2):
        num_points = 10  # Increase number of interpolated points for smoother lines
        
        # Draw a smooth line between two points using interpolation
        for i in range(1, num_points + 1):
            # Interpolating coordinates
            t = i / (num_points + 1)
            interp_x = int(x1 + (x2 - x1) * t)
            interp_y = int(y1 + (y2 - y1) * t)
            
            # Draw a smaller oval to make the stroke appear smoother
            self.canvas.create_oval(interp_x - 4, interp_y - 4, interp_x + 4, interp_y + 4, fill='black')
            self.draw.ellipse([interp_x - 4, interp_y - 4, interp_x + 4, interp_y + 4], fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill='white')
        self.result_label.config(text="Draw a digit!")
        self.last_x, self.last_y = None, None

    def is_canvas_empty(self):
        # Check if the image is completely white (no drawing has been made)
        img_array = np.array(self.image)
        return np.all(img_array == 255)  # Returns True if the image is all white

    def predict_digit(self):
        if self.is_canvas_empty():
            # Skip prediction if the canvas is empty
            return

        # Proceed with prediction if the canvas is not empty
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28)

        pred = model.predict(img_array)
        digit = np.argmax(pred)
        confidence = np.max(pred)

        # Update the dropdown to show the predicted label
        self.correct_label_var.set(str(digit))

        self.result_label.config(text=f"Prediction: {digit} ({confidence * 100:.2f}%)")

        # Show distribution in the live graph (side by side)
        self.update_graph(pred)

    def save_image(self):
        self.image.save("my_digit.png")
        self.result_label.config(text="Image saved as my_digit.png")

    def update_graph(self, predictions):
        self.graph.clear()  # Clear the current axes to prevent new figures from opening

        bars = self.graph.bar(range(10), predictions[0], color='skyblue')
        self.graph.set_xlabel("Digit")
        self.graph.set_ylabel("Probability")
        self.graph.set_title("Prediction Confidence")
        self.graph.set_xticks(range(10))

        # Add percentage labels
        for bar, prob in zip(bars, predictions[0]):
            self.graph.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{prob * 100:.1f}%', 
                            ha='center', va='bottom', fontsize=9)

        self.canvas_widget.draw()

    def learn_from_drawing(self):
        # Get the correct label from dropdown
        label = int(self.correct_label_var.get())
        
        # Preprocess the current drawing
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)

        # Crop the image to remove any white spaces and resize it
        bbox = img.getbbox()
        img = img.crop(bbox)
        img = img.resize((20, 20), Image.LANCZOS)
        new_img = Image.new("L", (28, 28), 0)
        new_img.paste(img, ((28 - 20) // 2, (28 - 20) // 2))

        img_array = np.array(new_img) / 255.0
        img_array = img_array.reshape(1, 28, 28)

        # Save the new data to a .npz file
        try:
            data = np.load("user_training_data.npz")
            images = np.concatenate([data["images"], img_array])
            labels = np.concatenate([data["labels"], [label]])
        except FileNotFoundError:
            images = img_array
            labels = np.array([label])

        np.savez("user_training_data.npz", images=images, labels=labels)

        self.result_label.config(text="Saved example for learning!")

# Create the Tkinter window and run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root)
    root.mainloop()
