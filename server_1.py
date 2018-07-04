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
import multiprocessing
from functools import partial

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
imgUrls = []

class FeatureThread(Thread):
	def __init__(self, name, *args):
		super(FeatureThread,self).__init__(name = name)
		self.data = args
		self.init = False

	def run(self):
		# print(self.name, self.data)
		while True:
			if not self.init:
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
			if self.init:
				print('refresh feature done, total feature:', len(features))
			else:
				print('load feature done, ready to serve, total feature:', len(features))
				self.init = True
			time.sleep(60)

FeatureThread("feature", range(10)).start()

def CalNorm(query, feature):
	return np.linalg.norm(feature - query)

cores = multiprocessing.cpu_count()
cores = cores * 2
pool = multiprocessing.Pool(processes=cores)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "_" + file.filename
        img.save(uploaded_img_path)

        query = fe.extract(img)
        startTime = time.time()

        dists = pool.map(partial(CalNorm, query), features)

        # dists = np.linalg.norm(features - query, axis=1)  # Do search
        ids = np.argsort(dists)[:48] # Top 48 results
        scores = [(dists[id], imgUrls[id]) for id in ids]
        endTime = time.time()
        app.logger.warning('Cost: %f', endTime - startTime)

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores, total=len(scores), f_total=len(features), cost=endTime - startTime)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8001, debug=False)