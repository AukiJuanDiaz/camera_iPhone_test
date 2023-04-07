from flask import Flask, render_template, request, send_file, redirect, url_for
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    # Get the uploaded image from the request
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Perform your calculations on the image
    # Here's an example that simply inverts the colors
    img = cv2.bitwise_not(img)

    # Save the processed image to a temporary file
    temp_file = 'static/temp.jpg'
    cv2.imwrite(temp_file, img)

    # Redirect to the processed image page
    return redirect(url_for('processed_image'))

@app.route('/processed_image')
def processed_image():
    # Render the HTML template with the URL of the processed image
    # Explain url_for() here
    return render_template('processed_image.html', image_url=url_for('static', filename='temp.jpg'))






if __name__ == '__main__':
    app.run(debug=True)
