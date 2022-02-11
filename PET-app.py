from flask import Flask, flash, request, redirect, render_template, url_for,\
    send_from_directory, abort
from werkzeug.utils import secure_filename
from PIL import Image
import face_recognition
import imghdr
import pickle
import glob
import os


app = Flask(__name__)

UPLOAD_FOLDER = 'C:/Users/olego/uploads/'
ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif']

app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

dir_rus_men = 'C:/Users/olego/static/images/PET/men/rus'
dir_rus_women = 'C:/Users/olego/static/images/PET/women/rus'
dir_eng_men = 'C:/Users/olego/static/images/PET/men/eng'
dir_eng_women = 'C:/Users/olego/static/images/PET/women/eng'

def image_resizer(img):
    SIZE = 320
    # загружаем при помощи face_recognition изображение
    image = face_recognition.load_image_file(img)
    # получаем координаты расположения лица (top-right, bottom-left)
    face_locations = face_recognition.face_locations(image)
    
    # если лицо не распозднается face_recognition, то удаляем это изображение
    if face_locations == []:
        os.remove(img)
        print('[%] Image: {} delete!'.format(img))
    else:
        face_width_coord_bigger = face_locations[0][1]
        face_width_coord_smaller = face_locations[0][3]
        
        face_height_coord_bigger = face_locations[0][2]
        face_height_coord_smaller = face_locations[0][0]
        
        face_width_frame = int((face_width_coord_bigger - face_width_coord_smaller) * 0.63)
        face_height_frame = int((face_height_coord_bigger - face_height_coord_smaller) * 0.63)
        
        # загрузка изображения
        image = Image.open(img)
        # получение размеров изображения
        size = image.size
        
        # получение ширины и высоты
        width = image.size[0]
        height = image.size[1]
        
        # вычисление новых координат для кропа
        # координата слева
        if face_width_coord_smaller - face_width_frame > 0:
            left = face_width_coord_smaller - face_width_frame
        else:
            left = 0
            
        # координата сверху
        if face_height_coord_smaller - face_height_frame > 0:
            top = face_height_coord_smaller - face_height_frame
        else:
            top = 0
            
        # координата справа
        if face_width_coord_bigger + face_width_frame < width:
            right = face_width_coord_bigger + face_width_frame
        else:
            right = width
            
        # координата снизу
        if face_height_coord_bigger + face_height_frame < height:
            bottom = face_height_coord_bigger + face_height_frame
        else:
            bottom = height
            
        # кроп изображения
        resized_image = image.crop((left, top, right, bottom))
        size = resized_image.size
        coef = SIZE / size[0]
        resized_image = resized_image.resize((int(size[0] * coef), int(size[1] * coef)))
        resized_image = resized_image.convert('RGB')
        resized_image.save(img)
        #return None

def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

@app.errorhandler(413)
def too_large(e):
    return "Файл слишком большой", 413

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)  

def get_embeding(path):
    face = face_recognition.load_image_file(path)
    # Преобразуем фото с лицом в вектор, получаем embeding   
    face_enc = face_recognition.face_encodings(face)
    return face_enc[0]
        
def predict_gender(face_enc, model):
        # Прогноз пола по обученной модели
        predict = model.predict([face_enc])
        return predict[0]

def sex_determination(face_enc):
    pkl_patch = 'C:/Users/olego/Documents/DataScience/PET-project/static/models/'
    with open(pkl_patch + 'model_gender_determination.pkl','rb') as f:
        model = pickle.load(f)
    return predict_gender(face_enc, model)

def predict_name(face_enc, model, dict_labels):
    predict = model.predict([face_enc])
    predict_label = list(dict_labels.keys())[list(dict_labels.values()).\
                                              index(predict)]
    return predict_label

def get_rus_similar_name(face_enc, gender):
    pkl_patch = 'C:/Users/olego/Documents/DataScience/PET-project/static/'
    if gender:
        with open(pkl_patch + 'rus-actresses-dict-labels.pkl','rb') as f:
            dict_labels = pickle.load(f)
        with open(pkl_patch + 'models/rus_actresses.pkl','rb') as f:
            model = pickle.load(f)
        rus_name = predict_name(face_enc, model, dict_labels)
    else:
        with open(pkl_patch + 'rus-actors-dict-labels.pkl','rb') as f:
            dict_labels = pickle.load(f)
        with open(pkl_patch + 'models/rus_actors.pkl','rb') as f:
            model = pickle.load(f)
        rus_name = predict_name(face_enc, model, dict_labels)
    return rus_name

def get_eng_similar_name(face_enc, gender):
    pkl_patch = 'C:/Users/olego/Documents/DataScience/PET-project/static/'
    if gender:
        with open(pkl_patch + 'eng-actresses-dict-labels.pkl','rb') as f:
            dict_labels = pickle.load(f)
        with open(pkl_patch + 'models/eng_actresses.pkl','rb') as f:
            model = pickle.load(f)
        eng_name = predict_name(face_enc, model, dict_labels)
    else:
        with open(pkl_patch + 'eng-actors-dict-labels.pkl','rb') as f:
            dict_labels = pickle.load(f)
        with open(pkl_patch + 'models/eng_actors.pkl','rb') as f:
            model = pickle.load(f)
        eng_name = predict_name(face_enc, model, dict_labels)
    return eng_name
    

@app.route('/', methods=['POST'])
def upload_file():
    # загрузка через форму
    if request.content_type != 'application/x-www-form-urlencoded':        
        file = request.files['file']
        filename = secure_filename(file.filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            #if file_ext not in app.config['ALLOWED_EXTENSIONS'] or \
            #    file_ext != validate_image(file.stream):
            if file_ext not in app.config['ALLOWED_EXTENSIONS']:
                return "Invalid image", 400
            else:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                image_resizer(file_path)
                # получение embeding для загруженого фото
                face_enc = get_embeding(file_path)
                # определение пола
                if sex_determination(face_enc):
                    # женщина
                    gender = 1
                    role = 'актрис'
                    rus_name = get_rus_similar_name(face_enc, gender)
                    eng_name = get_eng_similar_name(face_enc, gender)
                    img_files = glob.glob(dir_rus_women + '/' + rus_name + '/*')
                    images_rus = [img.split('\\')[-1] for img in img_files]
                    img_files = glob.glob(dir_eng_women + '/' + eng_name + '/*')
                    images_eng = [img.split('\\')[-1] for img in img_files]
                else:
                    # мужчина
                    gender = 0
                    role = 'актёров'
                    rus_name = get_rus_similar_name(face_enc, gender)
                    eng_name = get_eng_similar_name(face_enc, gender)
                    img_files = glob.glob(dir_rus_men + '/' + rus_name + '/*')
                    images_rus = [img.split('\\')[-1] for img in img_files]
                    img_files = glob.glob(dir_eng_men + '/' + eng_name + '/*')
                    images_eng = [img.split('\\')[-1] for img in img_files]
                
                return render_template('resp2.html', role=role, rus_name=rus_name,
                                       eng_name=eng_name, filename=filename,
                                       images_rus=images_rus, 
                                       images_eng=images_eng,
                                       gender=gender)
    # загрузка drag&drop
    else:
        fname = request.form['fname']
        if fname != '':
            file_ext = os.path.splitext(fname)[1]
            if file_ext not in app.config['ALLOWED_EXTENSIONS']:
                return "Invalid image", 400
            else:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                #file.save(file_path)
                
                image_resizer(file_path)         
                # получение embeding для загруженого фото
                face_enc = get_embeding(file_path)
                # определение пола
                if sex_determination(face_enc):
                    # женщина
                    gender = 1
                    role = 'актрис'
                    rus_name = get_rus_similar_name(face_enc, gender)
                    eng_name = get_eng_similar_name(face_enc, gender)
                    img_files = glob.glob(dir_rus_women + '/' + rus_name + '/*')
                    images_rus = [img.split('\\')[-1] for img in img_files]
                    img_files = glob.glob(dir_eng_women + '/' + eng_name + '/*')
                    images_eng = [img.split('\\')[-1] for img in img_files]
                else:
                    # мужчина
                    gender = 0
                    role = 'актёров'
                    rus_name = get_rus_similar_name(face_enc, gender)
                    eng_name = get_eng_similar_name(face_enc, gender)
                    img_files = glob.glob(dir_rus_men + '/' + rus_name + '/*')
                    images_rus = [img.split('\\')[-1] for img in img_files]
                    img_files = glob.glob(dir_eng_men + '/' + eng_name + '/*')
                    images_eng = [img.split('\\')[-1] for img in img_files]
                
                return render_template('resp2.html', role=role, rus_name=rus_name,
                                   eng_name=eng_name, filename=fname,
                                   images_rus=images_rus, 
                                   images_eng=images_eng,
                                   gender=gender)

    return '', 204

if __name__ == '__main__':
      app.run(host='127.0.0.1', port=80)