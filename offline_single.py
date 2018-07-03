import glob
import os
import pickle
from PIL import Image
from feature_extractor import FeatureExtractor
from threading import Thread
import threading
import time

fe = FeatureExtractor()

def GetFeature(imgPath):
	try:
		img = Image.open(imgPath)  # PIL image
		feature = fe.extract(img)
		feature_path = 'static/feature/' + os.path.splitext(os.path.basename(img_path))[0] + '.pkl'
		pickle.dump(feature, open(feature_path, 'wb'))
		pass
	except Exception as e:
		print(e)

idx = 0

startTime = time.time()

for img_path in sorted(glob.glob('static/img/*.jpg')):
	try:
		idx = idx+1
		nowTIme = time.time()
		detTime = nowTIme - startTime

		print(img_path, idx)
		if detTime > 0:
			print('qps', idx / detTime)
		GetFeature(img_path)
		pass
	except Exception as e:
		print(e)