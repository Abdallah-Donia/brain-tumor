from keras.models import load_model
model = load_model("modelFineT.h5")
image_size = (224,224)
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
def classify(img_path):
    img = keras.preprocessing.image.load_img(img_path, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return class_names[np.argmax(score)]

app = Flask(__name__)
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/classify_brain' , methods = ['POST'])
def classify_handler():
    img_path = request.form['path']
    y_pred = classify(img_path)
    print("img_path: " , img_path)
    print("Prediction: ", y_pred)
    return {"class_name" : y_pred}
app.run(port=5050)
