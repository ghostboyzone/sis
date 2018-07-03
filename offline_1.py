import glob
import os
import sys
import pickle
from PIL import Image
from feature_extractor import FeatureExtractor
from threading import Thread
import threading
import time
from urllib import request
import hashlib
import ssl

# 去掉ssl认证
ssl._create_default_https_context = ssl._create_unverified_context

fe = FeatureExtractor()

def GetFeature(imgUrl):
	try:
		response = request.urlopen(imgUrl, timeout=5.0)
		binaryData = response.read()
		uniqKey = Md5Sum(binaryData)
		imgPath = 'static/temp/' + uniqKey + '.jpg'
		imgFile = open(imgPath, 'wb')
		imgFile.write(binaryData)
		imgFile.close()

		img = Image.open(imgPath)  # PIL image
		feature = fe.extract(img)
		featurePath = 'static/feature_url/' + os.path.splitext(os.path.basename(imgPath))[0] + '.pkl'
		featureMapPath = 'static/feature_url/' + os.path.splitext(os.path.basename(imgPath))[0] + '.map'
		pickle.dump(feature, open(featurePath, 'wb'))
		mapPath = open(featureMapPath, 'w')
		mapPath.write(imgUrl)
		mapPath.close()
		os.remove(imgPath)
		print(featurePath, featureMapPath)
		pass
	except Exception as e:
		print(e)

def Md5Sum(binaryData):
    fmd5 = hashlib.md5(binaryData)
    return fmd5.hexdigest()

idx = 0
startTime = time.time()
f = open('static/img_url.txt')
for line in f:
	imgUrl = line.strip()
	idx = idx+1
	print(imgUrl, idx)
	GetFeature(imgUrl)
	nowTIme = time.time()
	detTime = nowTIme - startTime
	if detTime > 0:
		print('qps', idx / detTime)
	# sys.exit(0)
f.close()