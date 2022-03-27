import os
from flask import Flask, render_template, request, redirect, url_for, abort, Response
from werkzeug.utils import secure_filename

from utils import uniquify
from web.project import Project
from web.video_processing import gen_frames

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2056 * 2056 * 30 * 60* 15
app.config['UPLOAD_EXTENSIONS'] = ['.mp4']
app.config['UPLOAD_PATH'] = 'tmp'



@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)

        path = os.path.join(app.config['UPLOAD_PATH'], filename)
        project = Project(filename,path)
        file = project.get_file_path()
        uploaded_file.save(file)
    return redirect(url_for('index'))
app.run(debug=True)