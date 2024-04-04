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
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            web_path = url_for('static', filename=os.path.join('uploads', filename))
            output = get_top_placements(file_path)
            return render_template_string('''
                <!doctype html>
                <title>Upload new Image</title>
                <h1>Upload another image</h1>
                <form method=post enctype=multipart/form-data>
                  <input type=file name=file>
                  <input type=submit value=Upload>
                </form>
                <img src="{{ filepath }}" alt="uploaded image" width=400 />
                <p>{{ output.replace('\n', '<br>')|safe }}</p>  <!-- Display the output here -->
            ''', output=output, filepath=web_path)

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return f'File uploaded successfully: {filename}'

if __name__ == '__main__':
    app.run(debug=True)
