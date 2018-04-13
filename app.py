"""
	Many part of this app adapted from (https://github.com/experiencor/basic-yolo-keras)
	Modified to run on server via flask-api.

	Author:
		Nguyen Thanh An <annt@vng.com.vn>
"""

from __future__ import print_function

from flask import request, url_for, jsonify, send_file, Response, render_template
from flask_api import FlaskAPI, status, exceptions
from skimage import io as sk_io
import urllib2
import io
import base64

import argparse
import os
import cv2
import numpy as np
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO-v2 model on any dataset')
argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')
argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

# section for prediction
def load_model(args):
	config_path  = args.conf
	weights_path = args.weights

	with open(config_path) as config_buffer:
	    config = json.load(config_buffer)
	    
	###############################
	#   Make the model 
	###############################
	
	yolo = YOLO(architecture        = config['model']['architecture'],
	            input_size          = config['model']['input_size'], 
	            labels              = config['model']['labels'], 
	            max_box_per_image   = config['model']['max_box_per_image'],
	            anchors             = config['model']['anchors'])
	
	###############################
	#   Load trained weights
	###############################    
	
	print (weights_path)
	yolo.load_weights(weights_path)
	
	return yolo, config

def download_image(url):
	rgb_img = sk_io.imread(url) # rgb
	bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
	# img_str = urllib2.urlopen(url).read()
	# nparr = np.fromstring(img_str, np.uint8)
	# img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	return bgr_img
	
# def encode_image(image_path):
# 	image_content = image.read()
# 	return base64.b64encode(image_content)

yolo, config = load_model(argparser.parse_args())
labels = config['model']['labels']
def predict(bgr_img):
	image = bgr_img
	boxes = yolo.predict(image)
	image = draw_boxes(image, boxes, config['model']['labels'])

	return image, boxes

# section for API
app = FlaskAPI(__name__)

@app.route('/')
def api_root():
    return render_template('image_search.html')
    # app.send_static_file('image_search.html')

@app.route("/predict", methods=['GET', 'POST'])
def search():
	link = request.data.get('link', '').encode('utf-8')
    
	if not link:
		return jsonify({
			"success": False,
			"message": "Link to image not found.",
			"link": link}), status.HTTP_201_CREATED
	else:
	    try:
			image = download_image(link)
			image, boxes = predict(image)

			unique_labels, counts = np.unique([labels[box.get_label()] for box in boxes], return_counts=True)
			objects_count = dict(zip(unique_labels, counts))
			
			fname = link.split('/')[-1]
			path_out = os.path.join('detected_images', fname)
			cv2.imwrite(path_out, image)

			print ('[Request] link: {}, saved: {}, boxes: {}'.format(link, fname, len(boxes)))

			with open(path_out, 'rb') as image_buffer:
				encoded_str = base64.b64encode(image_buffer.read())
				return jsonify({
					"success": True,
					"img_encoded": encoded_str,
					"objects_count": objects_count}), status.HTTP_201_CREATED

	    except Exception as ex:
	        print ('[Error] {}'.format(ex))
	    	return jsonify({"success": False, "message": repr(ex)}), status.HTTP_201_CREATED


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=4000)

