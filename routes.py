from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from application import app
# from application import config
from flask import Flask,render_template, json, request, Response, flash,redirect, url_for, session
from flask_mysqldb import MySQL, MySQLdb
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
import mysql.connector
import urllib
import datetime
from selenium import webdriver
from mysql.connector import Error
from mysql.connector import errorcode
from tkinter import * 
from tkinter import messagebox
# from flask_ngrok import run_with_ngrok
from keras.models import load_model
from PIL import Image
import tensorflow.compat.v1 as tf
import random
import numpy as np
import pickle
import json
# from flask_ngrok import run_with_ngrok
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()



app = Flask(__name__)

app.secret_key = "secretkeytuing"

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'bigprojeck'

Mysql = MySQL(app)
# classes = pickle.load(open("classes.pkl", "rb"))
# run_with_ngrok(app) 

@app.route("/")
def main():
    return render_template('login.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/absensimhs')
def absensimhs():
    # cur = Mysql.connection.cursor()
    # cur.execute("SELECT * FROM jadwal")
    # jadwal = cur.fetchall()
    # cur.close()
    return render_template('absensimhs.html')

@app.route('/rekapabsen')
def rekapabsen():
    # cur = Mysql.connection.cursor()
    # cur.execute("SELECT * FROM jadwal")
    # jadwal = cur.fetchall()
    # cur.close()
    return render_template('rekapabsen.html')


@app.route('/jadwala')
def jadwala():
    cur = Mysql.connection.cursor()
    cur.execute("SELECT * FROM jadwal")
    jadwal = cur.fetchall()
    cur.close()
    return render_template('jadwala.html',data = jadwal)

@app.route('/jadwalb')
def jadwalb():
    cur = Mysql.connection.cursor()
    cur.execute("SELECT * FROM jadwal")
    jadwal = cur.fetchall()
    cur.close()
    return render_template('jadwalb.html',data = jadwal)


@app.route('/jadwalc')
def jadwalc():
    cur = Mysql.connection.cursor()
    cur.execute("SELECT * FROM jadwal")
    jadwal = cur.fetchall()
    cur.close()
    return render_template('jadwalc.html',data = jadwal)

@app.route('/jadwald')
def jadwald():
    cur = Mysql.connection.cursor()
    cur.execute("SELECT * FROM jadwal")
    jadwal = cur.fetchall()
    cur.close()
    return render_template('jadwald.html',data = jadwal)

@app.route('/jadwale')
def jadwale():
    cur = Mysql.connection.cursor()
    cur.execute("SELECT * FROM jadwal")
    jadwal = cur.fetchall()
    cur.close()
    return render_template('jadwale.html',data = jadwal)

@app.route('/rekapa')
def rekapa():
    cur = Mysql.connection.cursor()
    cur.execute("SELECT * FROM kelasa")
    jadwal = cur.fetchall()
    cur.close()
    return render_template('rekapa.html',data = jadwal)

@app.route('/rekapb')
def rekapb():
    cur = Mysql.connection.cursor()
    cur.execute("SELECT * FROM kelasb")
    jadwal = cur.fetchall()
    cur.close()
    return render_template('rekapb.html',data = jadwal)

@app.route('/rekapc')
def rekapc():
    cur = Mysql.connection.cursor()
    cur.execute("SELECT * FROM kelasc")
    jadwal = cur.fetchall()
    cur.close()
    return render_template('rekapc.html',data = jadwal)

@app.route('/rekapd')
def rekapd():
    cur = Mysql.connection.cursor()
    cur.execute("SELECT * FROM kelasd")
    jadwal = cur.fetchall()
    cur.close()
    return render_template('rekapd.html',data = jadwal)

@app.route('/rekape')
def rekape():
    cur = Mysql.connection.cursor()
    cur.execute("SELECT * FROM kelase")
    jadwal = cur.fetchall()
    cur.close()
    return render_template('rekape.html',data = jadwal)

@app.route('/rekapf')
def rekapf():
    cur = Mysql.connection.cursor()
    cur.execute("SELECT * FROM kelasf")
    jadwal = cur.fetchall()
    cur.close()
    return render_template('rekapf.html',data = jadwal)

@app.route('/bantuan',methods=["GET", "POST"])
def bantuan():

    return render_template('bantuan.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        nim = request.form['nim']
        password = request.form['password']
        curl = Mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        curl.execute("SELECT * FROM user WHERE nim=%s AND password=%s", (nim,password) )
        user = curl.fetchone()
        curl.close()

        if user is not None and len(user) > 0:
            if user['password'] == user['password']:
                session['nim'] = user['nim']
                session['password'] = user['password']
                flash('login berhasil', 'success')
                return redirect(url_for('index'))
            else:
                flash("Gagal, nim dan password tidak cocok")
                return redirect(url_for('login'))
        else:
            flash("Gagal, user tidak ditemukan")
            return redirect(url_for('login'))
    else:
        return render_template('login.html')

@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    else:
        nim = request.form['nim']
        nama = request.form['nama']
        email = request.form['email']
        password = request.form['password']
        # hash_password = bcrypt.hashpw(password, bcrypt.gensalt())

        cur = Mysql.connection.cursor()
        cur.execute("INSERT INTO user (nim, nama, email, password) VALUES (%s,%s,%s,%s)",(nim,nama,email,password))
        Mysql.connection.commit()
        flash('Pendaftaran Berhasil', 'success')
        session['nim'] = request.form['nim']
        session['password'] = request.form['password']
        return redirect(url_for('login'))

@app.route('/verifwajah')
def verifwajah():
    return render_template('verifwajah.html')

@app.route('/absenkelasa')
def absenkelasa():
    return render_template('absenkelasa.html')

@app.route('/absenkelasb')
def absenkelasb():
    return render_template('absenkelasb.html')

@app.route('/absenkelasc')
def absenkelasc():
    return render_template('absenkelasc.html')

@app.route('/absenkelasd')
def absenkelasd():
    return render_template('absenkelasd.html')

@app.route('/riwayat')
def riwayat():
    return render_template('riwayat.html')

@app.route('/pengaturanakun',methods = ['POST', 'GET'])
def profil():
    # cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    # cur.execute('SELECT * FROM user WHERE nim = %s', (nim))
    # data = cur.fetchall()
    # cur.close()
    # print(data[0])
    return render_template('pengaturanakun.html')

# @app.route('/pengaturanakun/<nim>',methods = ['POST', 'GET'])
# def profil(nim):
#     cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
#     cur.execute('SELECT * FROM user WHERE nim = %s', (nim))
#     data = cur.fetchall()
#     cur.close()
#     print(data[0])
#     return render_template('pengaturanakun.html', data = data[0])
    

# @app.route('/updateakun/<nim>', methods=['POST'])
# def updateService(nim):
#     if request.method == 'POST':
#         nama = request.form['nama']
#         email = request.form['email']
#         password = request.form['password']
#         cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
#         cur.execute("""
#             UPDATE user
#             SET nama = %s,
#                 email = %s,
#                 password = %s
#             WHERE nim = %s
#         """, (nama, email, password, nim))
#         flash('Data sudah Terupdate')
#         mysql.connection.commit()
#         return redirect(url_for('profil'))



                
#         ret, buffer = cv2.imencode('.jpg', img)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl' 
npy='./npy'
train_img="./train_img"

@app.route('/gen_frames1')
def gen_frames1():
    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
            minsize = 30  # minimum size of face
            threshold = [0.7,0.8,0.8]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            batch_size =100 #1000
            image_size = 182
            input_image_size = 160
            HumanNames = os.listdir(train_img)
            HumanNames.sort()
            print('Loading Model')
            facenet.load_model(modeldir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile,encoding='latin1')

            video_capture = cv2.VideoCapture(0)
            print('Start Recognition')
            while True:
                ret, frame = video_capture.read()
                #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
                timer =time.time()
                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                faceNum = bounding_boxes.shape[0]
                if faceNum > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]
                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    for i in range(faceNum):
                        emb_array = np.zeros((1, embedding_size))
                        xmin = int(det[i][0])
                        ymin = int(det[i][1])
                        xmax = int(det[i][2])
                        ymax = int(det[i][3])
                        try:
                            # inner exception
                            if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                                print('Face is very close!')
                                continue
                            cropped.append(frame[ymin:ymax, xmin:xmax,:])
                            cropped[i] = facenet.flip(cropped[i], False)
                            scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                    interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            if best_class_probabilities>0.80:
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)    #boxing face
                                for H_i in HumanNames:
                                    if HumanNames[best_class_indices[0]] == H_i:
                                        result_names = HumanNames[best_class_indices[0]]
                                        print("Predictions : [ name: {} , accuracy: {:.3f} ]".format(HumanNames[best_class_indices[0]],best_class_probabilities[0]))
                                        cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                                        cv2.putText(frame, result_names, (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, 0, 0), thickness=1, lineType=1)

                                video_capture.release()
                                cv2.destroyAllWindows()
                                # driver = webdriver.Firefox()
                                # driver.refresh()
                                import webbrowser
                                webbrowser.open_new('http://127.0.0.1:5000/absenkelasa')

                                        
                                        
                            else :
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                                cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                                cv2.putText(frame, "Tidak Terdefinisi", (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, 0, 0), thickness=1, lineType=1)
                        except:   
                            
                            print("error")


                        ct = datetime.datetime.now()
                        try:
                                        connection = mysql.connector.connect(host='localhost',
                                                                            database='bigprojeck',
                                                                            user='root',
                                                                            password='')
                                        

                                        cursor = connection.cursor()
                                        cursor.execute("INSERT INTO kelasa VALUES (%s, %s)", (result_names,ct))
                                        connection.commit()
                                        print(cursor.rowcount, "data sudah masuk")
                                        cursor.close()

                        except mysql.connector.Error as error:
                                                        print("tidak dapat input data {}".format(error))

                        finally:
                            if (connection.is_connected()):
                                        connection.close()
                                        print("MySQL connection is closed")
                                        break
                        
                endtimer = time.time()
                fps = 1/(endtimer-timer)
                cv2.rectangle(frame,(15,30),(135,60),(0,255,255),-1)
                cv2.putText(frame, "fps: {:.2f}".format(fps), (20, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frames = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frames + b'\r\n')    
        #     cv2.imshow('Face Recognition', frame)
        #     key= cv2.waitKey(1)
        #     if key== 113: # "q"
        #         break
        # video_capture.release()
        # cv2.destroyAllWindows()

@app.route('/gen_frames2')
def gen_frames2():
    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
            minsize = 30  # minimum size of face
            threshold = [0.7,0.8,0.8]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            batch_size =100 #1000
            image_size = 182
            input_image_size = 160
            HumanNames = os.listdir(train_img)
            HumanNames.sort()
            print('Loading Model')
            facenet.load_model(modeldir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile,encoding='latin1')

            video_capture = cv2.VideoCapture(0)
            print('Start Recognition')
            while True:
                ret, frame = video_capture.read()
                #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
                timer =time.time()
                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                faceNum = bounding_boxes.shape[0]
                if faceNum > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]
                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    for i in range(faceNum):
                        emb_array = np.zeros((1, embedding_size))
                        xmin = int(det[i][0])
                        ymin = int(det[i][1])
                        xmax = int(det[i][2])
                        ymax = int(det[i][3])
                        try:
                            # inner exception
                            if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                                print('Face is very close!')
                                continue
                            cropped.append(frame[ymin:ymax, xmin:xmax,:])
                            cropped[i] = facenet.flip(cropped[i], False)
                            scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                    interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            if best_class_probabilities>0.80:
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)    #boxing face
                                for H_i in HumanNames:
                                    if HumanNames[best_class_indices[0]] == H_i:
                                        result_names = HumanNames[best_class_indices[0]]
                                        print("Predictions : [ name: {} , accuracy: {:.3f} ]".format(HumanNames[best_class_indices[0]],best_class_probabilities[0]))
                                        cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                                        cv2.putText(frame, result_names, (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, 0, 0), thickness=1, lineType=1)

                                video_capture.release()
                                cv2.destroyAllWindows()
                                # driver = webdriver.Firefox()
                                # driver.refresh()
                                import webbrowser
                                webbrowser.open_new('http://127.0.0.1:5000/absenkelasb')

                                        
                                        
                            else :
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                                cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                                cv2.putText(frame, "Tidak Terdefinisi", (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, 0, 0), thickness=1, lineType=1)
                        except:   
                            
                            print("error")


                        ct = datetime.datetime.now()
                        try:
                                        connection = mysql.connector.connect(host='localhost',
                                                                            database='bigprojeck',
                                                                            user='root',
                                                                            password='')
                                        

                                        cursor = connection.cursor()
                                        cursor.execute("INSERT INTO kelasb VALUES (%s, %s)", (result_names,ct))
                                        connection.commit()
                                        print(cursor.rowcount, "data sudah masuk")
                                        cursor.close()

                        except mysql.connector.Error as error:
                                                        print("tidak dapat input data {}".format(error))

                        finally:
                            if (connection.is_connected()):
                                        connection.close()
                                        print("MySQL connection is closed")
                                        break
                        
                endtimer = time.time()
                fps = 1/(endtimer-timer)
                cv2.rectangle(frame,(15,30),(135,60),(0,255,255),-1)
                cv2.putText(frame, "fps: {:.2f}".format(fps), (20, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frames = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frames + b'\r\n')    
        #     cv2.imshow('Face Recognition', frame)
        #     key= cv2.waitKey(1)
        #     if key== 113: # "q"
        #         break
        # video_capture.release()
        # cv2.destroyAllWindows()

@app.route('/gen_frames3')
def gen_frames3():
    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
            minsize = 30  # minimum size of face
            threshold = [0.7,0.8,0.8]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            batch_size =100 #1000
            image_size = 182
            input_image_size = 160
            HumanNames = os.listdir(train_img)
            HumanNames.sort()
            print('Loading Model')
            facenet.load_model(modeldir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile,encoding='latin1')

            video_capture = cv2.VideoCapture(0)
            print('Start Recognition')
            while True:
                ret, frame = video_capture.read()
                #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
                timer =time.time()
                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                faceNum = bounding_boxes.shape[0]
                if faceNum > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]
                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    for i in range(faceNum):
                        emb_array = np.zeros((1, embedding_size))
                        xmin = int(det[i][0])
                        ymin = int(det[i][1])
                        xmax = int(det[i][2])
                        ymax = int(det[i][3])
                        try:
                            # inner exception
                            if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                                print('Face is very close!')
                                continue
                            cropped.append(frame[ymin:ymax, xmin:xmax,:])
                            cropped[i] = facenet.flip(cropped[i], False)
                            scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                    interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            if best_class_probabilities>0.80:
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)    #boxing face
                                for H_i in HumanNames:
                                    if HumanNames[best_class_indices[0]] == H_i:
                                        result_names = HumanNames[best_class_indices[0]]
                                        print("Predictions : [ name: {} , accuracy: {:.3f} ]".format(HumanNames[best_class_indices[0]],best_class_probabilities[0]))
                                        cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                                        cv2.putText(frame, result_names, (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, 0, 0), thickness=1, lineType=1)

                                video_capture.release()
                                cv2.destroyAllWindows()
                                # driver = webdriver.Firefox()
                                # driver.refresh()
                                import webbrowser
                                webbrowser.open_new('http://127.0.0.1:5000/absenkelasc')

                                        
                                        
                            else :
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                                cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                                cv2.putText(frame, "Tidak Terdefinisi", (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, 0, 0), thickness=1, lineType=1)
                        except:   
                            
                            print("error")


                        ct = datetime.datetime.now()
                        try:
                                        connection = mysql.connector.connect(host='localhost',
                                                                            database='bigprojeck',
                                                                            user='root',
                                                                            password='')
                                        

                                        cursor = connection.cursor()
                                        cursor.execute("INSERT INTO kelasc VALUES (%s, %s)", (result_names,ct))
                                        connection.commit()
                                        print(cursor.rowcount, "data sudah masuk")
                                        cursor.close()

                        except mysql.connector.Error as error:
                                                        print("tidak dapat input data {}".format(error))

                        finally:
                            if (connection.is_connected()):
                                        connection.close()
                                        print("MySQL connection is closed")
                                        break
                        
                endtimer = time.time()
                fps = 1/(endtimer-timer)
                cv2.rectangle(frame,(15,30),(135,60),(0,255,255),-1)
                cv2.putText(frame, "fps: {:.2f}".format(fps), (20, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frames = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frames + b'\r\n')    
        #     cv2.imshow('Face Recognition', frame)
        #     key= cv2.waitKey(1)
        #     if key== 113: # "q"
        #         break
        # video_capture.release()
        # cv2.destroyAllWindows()

@app.route('/gen_frames4')
def gen_frames4():
    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
            minsize = 30  # minimum size of face
            threshold = [0.7,0.8,0.8]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            batch_size =100 #1000
            image_size = 182
            input_image_size = 160
            HumanNames = os.listdir(train_img)
            HumanNames.sort()
            print('Loading Model')
            facenet.load_model(modeldir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile,encoding='latin1')

            video_capture = cv2.VideoCapture(0)
            print('Start Recognition')
            while True:
                ret, frame = video_capture.read()
                #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
                timer =time.time()
                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                faceNum = bounding_boxes.shape[0]
                if faceNum > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]
                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    for i in range(faceNum):
                        emb_array = np.zeros((1, embedding_size))
                        xmin = int(det[i][0])
                        ymin = int(det[i][1])
                        xmax = int(det[i][2])
                        ymax = int(det[i][3])
                        try:
                            # inner exception
                            if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                                print('Face is very close!')
                                continue
                            cropped.append(frame[ymin:ymax, xmin:xmax,:])
                            cropped[i] = facenet.flip(cropped[i], False)
                            scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                    interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            if best_class_probabilities>0.80:
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)    #boxing face
                                for H_i in HumanNames:
                                    if HumanNames[best_class_indices[0]] == H_i:
                                        result_names = HumanNames[best_class_indices[0]]
                                        print("Predictions : [ name: {} , accuracy: {:.3f} ]".format(HumanNames[best_class_indices[0]],best_class_probabilities[0]))
                                        cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                                        cv2.putText(frame, result_names, (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, 0, 0), thickness=1, lineType=1)

                                video_capture.release()
                                cv2.destroyAllWindows()
                                # driver = webdriver.Firefox()
                                # driver.refresh()
                                import webbrowser
                                webbrowser.open_new('http://127.0.0.1:5000/absenkelasd')

                                        
                                        
                            else :
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                                cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                                cv2.putText(frame, "Tidak Terdefinisi", (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, 0, 0), thickness=1, lineType=1)
                        except:   
                            
                            print("error")


                        ct = datetime.datetime.now()
                        try:
                                        connection = mysql.connector.connect(host='localhost',
                                                                            database='bigprojeck',
                                                                            user='root',
                                                                            password='')
                                        

                                        cursor = connection.cursor()
                                        cursor.execute("INSERT INTO kelasd VALUES (%s, %s)", (result_names,ct))
                                        connection.commit()
                                        print(cursor.rowcount, "data sudah masuk")
                                        cursor.close()

                        except mysql.connector.Error as error:
                                                        print("tidak dapat input data {}".format(error))

                        finally:
                            if (connection.is_connected()):
                                        connection.close()
                                        print("MySQL connection is closed")
                                        break
                        
                endtimer = time.time()
                fps = 1/(endtimer-timer)
                cv2.rectangle(frame,(15,30),(135,60),(0,255,255),-1)
                cv2.putText(frame, "fps: {:.2f}".format(fps), (20, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frames = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frames + b'\r\n')    
        #     cv2.imshow('Face Recognition', frame)
        #     key= cv2.waitKey(1)
        #     if key== 113: # "q"
        #         break
        # video_capture.release()
        # cv2.destroyAllWindows()

  

@app.route('/video_feed1')
def video_feed1():
    return Response(gen_frames1(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(gen_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed3')
def video_feed3():
    return Response(gen_frames3(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed4')
def video_feed4():
    return Response(gen_frames4(), mimetype='multipart/x-mixed-replace; boundary=frame')

# def predict_class(sentence, model):
#     # filter out predictions below a threshold
#     res = model.predict(np.array([p]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     # sort by strength of probability
#     results.sort(key=lambda x: x[1], reverse=True)
#     return_list = []
#     for r in results:
#         return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
#     return return_list

model = load_model("chatbot_model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# run_with_ngrok(app) 

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")


@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    if msg.startswith('my name is'):
        name = msg[11:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res =res1.replace("{n}",name)
    elif msg.startswith('hi my name is'):
        name = msg[14:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res =res1.replace("{n}",name)
    else:
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
    return res


# chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result




if __name__ == '__main__':
    app.run()