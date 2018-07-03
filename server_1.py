import os
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
import glob
import pickle
from datetime import datetime
from flask import Flask, request, render_template
import time
import sys
from threading import Thread
import copy

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
imgUrls = []

class FeatureThread(Thread):
	def __init__(self, name, *args):
		super(FeatureThread,self).__init__(name = name)
		self.data = args

	def run(self):
		# print(self.name, self.data)
		while True:
			print('load feature start')
			featuresLoop = []
			imgUrlsLoop = []
			for feature_path in glob.glob("static/feature_url/*.pkl"):
			    imgFileName = 'static/feature_url/' + os.path.splitext(os.path.basename(feature_path))[0] + '.map'
			    f = open(imgFileName)
			    imgUrl = f.read().strip()
			    f.close()
			    if len(imgUrl) == 0:
			    	continue
			    featuresLoop.append(pickle.load(open(feature_path, 'rb')))
			    imgUrlsLoop.append(imgUrl)
			global features
			global imgUrls
			features = copy.deepcopy(featuresLoop)
			imgUrls = copy.deepcopy(imgUrlsLoop)
			# print(imgUrls)
			print('load feature done, sleep for 30 seconds')
			time.sleep(30)

FeatureThread("feature", range(10)).start()
# sys.exit(0)

# for feature_path in glob.glob("static/feature_url/*.pkl"):
#     imgFileName = 'static/feature_url/' + os.path.splitext(os.path.basename(feature_path))[0] + '.map'
#     f = open(imgFileName)
#     imgUrl = f.read().strip()
#     f.close()
#     if len(imgUrl) == 0:
#     	continue
#     features.append(pickle.load(open(feature_path, 'rb')))
#     imgUrls.append(imgUrl)
#     # sys.exit(0)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "_" + file.filename
        img.save(uploaded_img_path)

        # print(features, imgUrls)

        startTime = time.time()
        query = fe.extract(img)
        dists = np.linalg.norm(features - query, axis=1)  # Do search
        ids = np.argsort(dists)[:30] # Top 30 results
        scores = [(dists[id], imgUrls[id]) for id in ids]

        endTIme = time.time()
        app.logger.warning('Cost: %f', endTIme - startTime)

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
