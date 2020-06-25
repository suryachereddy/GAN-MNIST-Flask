from flask import Flask, render_template, send_file,Response
import numpy as np
import base64
import tensorflow as tf 
from PIL import Image
import io  
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

from matplotlib.figure import Figure


app = Flask(__name__)
generator = tf.keras.models.load_model('gen_model') 



@app.route("/")
def index():
    return render_template('index.html')

@app.route("/<int:num>/generate.png")
def generate(num=None):
    
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


def create_figure():
    generator = tf.keras.models.load_model('gen_model') 
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    generated_image=generated_image[0,:,:,0]


    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = generated_image[:,0]
    ys = generated_image[0,:]
    axis.imshow(generated_image, cmap='gray')
    return fig


if __name__ == '__main__':
    app.run(debug=True)

