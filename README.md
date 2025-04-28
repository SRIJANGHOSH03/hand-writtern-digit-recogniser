**Handwritten Digit Recognizer and Learner**


This project is an interactive Python application that allows users to draw digits on a canvas, recognize the drawn digit using a pre-trained machine learning model, and teach the system new examples by saving drawings for further training.


Features


🖌️ Draw digits (0-9) freely on a canvas.



🤖 Recognize the drawn digit using a trained scikit-learn model.



📚 Save your own drawings with labels to expand the learning dataset.



📂 Data is persistently stored in a .npz file for reuse and future training.



Installation


Clone the repository:



bash

Copy

Edit

git clone https://github.com/yourusername/your-repo-name.git

cd your-repo-name

Install the required packages:



bash

Copy

Edit

pip install numpy scikit-learn opencv-python tkinter

Note: tkinter usually comes pre-installed with Python, but if needed, install it according to your system.


How to Run


Run the main Python file:


bash

Copy

Edit

python main.py

This will open a window where you can:



Draw a digit using your mouse.



Predict the digit by clicking the "Predict" button.


Clear the canvas with the "Clear" button.



Teach the system by entering a label (0-9) and clicking "Learn from Drawing."



The drawn images and their labels are saved automatically in user_training_data.npz.



File Structure

bash

Copy

Edit

/your-repo-name


  ├── main.py               # Main application file
  
  ├── user_training_data.npz # (Auto-created) Saved drawings and labels
  
  └── README.md              # Project documentation
  
How Data Saving Works

On each "Learn from Drawing" action:


The current drawing and its label are saved.


If user_training_data.npz already exists, the new data is appended to the existing data.


This ensures the model can be retrained later with a growing dataset of user-provided examples.



Future Improvements


Add a retraining feature to fine-tune the model with user-provided data.

Enhance the UI to allow editing or deleting saved drawings.

Support multiple digits in a single drawing.
