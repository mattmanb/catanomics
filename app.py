import os
from flask import Flask, request, redirect, url_for, render_template, render_template_string
from werkzeug.utils import secure_filename
from top_placements import *

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename("uploaded_image.jpg")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            try:
                output = get_top_placements(file_path)
                web_path = "./static/uploads/hom_img.jpg" # url_for('static', filename=os.path.join('uploads', "hom_img.jpg"))
                return render_template_string('''
                    <!doctype html>
                        <html lang="en">
                        <head>
                            <meta charset="UTF-8">
                            <meta name="viewport" content="width=device-width, initial-scale=1.0">
                            <title>Upload New Image</title>
                            <link rel="stylesheet" href="./static/styles.css"> <!-- Ensure the same CSS file is used -->
                        </head>
                        <body>
                            <div class="container">
                                <h1>Upload Another Catan Board</h1>
                                <p>Things to make sure of when uploading an image:</p>
                                <ul>
                                    <li>Upload an image before settlements/roads are placed</li>
                                    <li>Ensure numbers are placed in the center of each hex</li>
                                    <li>Take the image from a top-down perspective</li>
                                    <li>Try to reduce glare in the image</li>
                                    <li>Note: if the board is set up incorrectly (wrong amount of hexes/numbers) an error will be thrown</li>
                                    <li>A sample image is shown on the homepage</li>
                                </ul>
                                <form method="post" enctype="multipart/form-data" class="upload-form">
                                    <label for="file" class="file-label">Choose a file</label>
                                    <input type="file" name="file" id="file" class="file-input">
                                    <button type="submit" class="submit-btn">Upload</button>
                                </form>
                                {% if filename %}
                                    <p>Uploaded file: {{ filename }}</p>
                                    <p class="green">Catan board successfully processed! Look below for the top 10 starting positions.</p>
                                {% endif %}
                                <div class="results-section">
                                    <img src="{{ filepath }}" alt="Uploaded Image" class="uploaded-image" />
                                    <div class="top-spots">
                                        <p>{{ output.replace('\n', '<br>')|safe }}</p> <!-- Display the output here -->
                                    </div>
                                </div>
                            </div>
                        </body>
                        </html>
                ''', output=output, filepath=web_path, filename = file.filename)
            except Exception as e:
                output = "Error, try uploading a new image with the entire board within frame"
                print(f"Error: {e}")
                web_path = "./static/uploads/error.jpg" # url_for('static', filename=os.path.join('uploads', "error.jpg"))
                return render_template_string('''
                    <!doctype html>
                        <html lang="en">
                        <head>
                            <meta charset="UTF-8">
                            <meta name="viewport" content="width=device-width, initial-scale=1.0">
                            <title>Upload New Image</title>
                            <link rel="stylesheet" href="./static/styles.css"> <!-- Ensure the same CSS file is used -->
                        </head>
                        <body>
                            <div class="container">
                                <h1>Upload Another Catan Board</h1>
                                <p>Things to make sure of when uploading an image:</p>
                                <ul>
                                    <li>Upload an image before settlements/roads are placed</li>
                                    <li>Ensure numbers are placed in the center of each hex</li>
                                    <li>Take the image from a top-down perspective</li>
                                    <li>Try to reduce glare in the image</li>
                                    <li>Note: if the board is set up incorrectly (wrong amount of hexes/numbers) an error will be thrown</li>
                                    <li>A sample image is shown on the homepage</li>
                                </ul>
                                <form method="post" enctype="multipart/form-data" class="upload-form">
                                    <label for="file" class="file-label">Choose a file</label>
                                    <input type="file" name="file" id="file" class="file-input">
                                    <button type="submit" class="submit-btn">Upload</button>
                                </form>
                                {% if filename %}
                                    <p>Uploaded file: {{ filename }}</p>
                                    <p class="red">Error with uploaded image. Ensure the entire board is in frame, and the board is set up correctly.</p>
                                {% endif %}
                                <div class="results-section">
                                    <img src="{{ filepath }}" alt="Uploaded Image" class="uploaded-image" />
                                    <div class="top-spots">
                                        <p>{{ output.replace('\n', '<br>')|safe }}</p> <!-- Display the output here -->
                                    </div>
                                </div>
                            </div>
                        </body>
                        </html>
                ''', output=output, filepath=web_path, filename = file.filename)
            

    return '''
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload New File</title>
        <link rel="stylesheet" href="./static/styles.css"> <!-- Link to the CSS file -->
    </head>
    <body>
        <div class="container">
            <h1>Upload Catan Board</h1>
            <p>Things to make sure of when uploading an image:</p>
            <ul>
                <li>Upload an image before settlements/roads are placed</li>
                <li>Ensure numbers are placed in the center of each hex</li>
                <li>Take the image from a top-down perspective</li>
                <li>Try to reduce glare in the image</li>
                <li>Note: if the board is set up incorrectly (wrong amount of hexes/numbers) an error will be thrown</li>
                <li>A sample image is shown below:</li>
            </ul>
            <img src="./static/uploads/sample_img.jpeg" alt="sample upload image" class=sample-image>
            <form method="post" enctype="multipart/form-data" class="upload-form">
                <label for="file" class="file-label">Choose a file</label>
                <input type="file" name="file" id="file" class="file-input">
                <button type="submit" class="submit-btn">Upload</button>
            </form>
        </div>
    </body>
    </html>

    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return f'File uploaded successfully: {filename}'

if __name__ == '__main__':
    app.run(debug=True)


# Final changed made April 30th, 2024 at 9:39PM