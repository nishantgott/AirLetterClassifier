->This program uses python to capture an image using your camera and recognizes the alphabet you draw 
with your index finger.

->I used a convolutional neural network and trained it myself. It gave 98% + accuracy on test data.
I saved this model in a .h5 file which I will be using in my main program.

->openCV library is used to capture the image, process it into the format required to pass it 
as input to the neural network.
plt.imshow is used to show how the input data to the neural network looks for better understanding.

->How to use
 After downloading all the required dependencies, run the program.
 Two windows should appear , one on top of another.

 One window displays the video output along with your drawing.
 The other window displays only your drawings on a black canvas.

Long Press R => Reset the canvas (erase)
Long Press P => Predict the alphabet which you drew using your finger 
Long Press Q => Close all the windows and exit the program

->How to draw
 When only index finger is up, drawing mode is activated. Pixels will be drawn at the tip of the index finger.
 When middle finger is up, drawing mode is deactivated. You can use it to move your finger around without drawing anything.