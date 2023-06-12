#!/usr/bin/env python3
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils

import argparse
import sys
#import cv2

from segnet_utils import *

# parse the command line
parser = argparse.ArgumentParser(description="Segment a live camera stream using an semantic segmentation DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.segNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="fcn-resnet18-voc", help="pre-trained model to load, see below for options")
parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
parser.add_argument("--visualize", type=str, default="overlay,mask", help="Visualization options (can be 'overlay' 'mask' 'overlay,mask'")
parser.add_argument("--ignore-class", type=str, default="void", help="optional name of class to ignore in the visualization results (default: 'void')")
parser.add_argument("--alpha", type=float, default=150.0, help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 150.0)")
parser.add_argument("--stats", action="store_true", help="compute statistics about segmentation mask class output")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the segmentation network
net = jetson.inference.segNet(opt.network, sys.argv)

# set the alpha blending value
net.SetOverlayAlpha(opt.alpha)

# create video output
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)

# create buffer manager
buffers = segmentationBuffers(net, opt)

# create video source
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)

font = jetson.utils.cudaFont()

##################
#parameter
BGtolerance = 18
BRtolerance = 38
Bmintolerance=40
Bmaxtolerance=85
BminHeight = 30
Gmintolerance=4
##################

# process frames until user exits
while True:
	# capture the next image
	img_input = input.Capture()
	buffers.Alloc(img_input.shape, img_input.format)
	img_copy = jetson.utils.cudaAllocMapped(width=img_input.width, height=img_input.height, format=img_input.format)
	img_small = jetson.utils.cudaAllocMapped(width=buffers.mask.width, height=buffers.mask.height, format=buffers.mask.format)

	jetson.utils.cudaMemcpy(img_copy, img_input)
	jetson.utils.cudaResize(img_copy, img_small)

	# allocate buffers for this size image
	buffers.Alloc(img_input.shape, img_input.format)

	# process the segmentation network
	net.Process(img_input, ignore_class=opt.ignore_class)

	# generate the overlay
	if buffers.overlay:
		net.Overlay(buffers.overlay, filter_mode=opt.filter_mode)

	# generate the mask
	if buffers.mask:
		net.Mask(buffers.mask, filter_mode=opt.filter_mode)
		net.Mask(img_copy, filter_mode=opt.filter_mode)

        ###############################		
        #check each color
	if(1):
		print("img_copy.shape = ", img_copy.shape)
		RmaxX = 0
		RmaxY = 0
		RminX = img_copy.shape[1]
		RminY = img_copy.shape[0]
		GmaxX = 0
		GmaxY = 0
		GminX = img_copy.shape[1]
		GminY = img_copy.shape[0]
		BmaxX = 0
		BmaxY = 0
		BminX = img_copy.shape[1]
		BminY = img_copy.shape[0]
		seeRed = 0
		seeGreen = 0
		seeBlue = 0
		#### tip of the Green bone
		GtipX = -1
		GtipY = -1

		for x in range(img_copy.shape[1]):
			for y in range(img_copy.shape[0]):
				if(img_copy[y,x]==(255,0,0)): # red color
					seeRed = 1
					if(x<RminX):
						RminX = x
					if(y<RminY):
						RminY = y
					if(x>RmaxX):
						RmaxX = x
					if(y>RmaxY):
						RmaxY = y
				if(img_copy[y,x]==(0,255,0)): # green color
					seeGreen = 1
					if(x>50):
					    if(x<GminX):
						    GminX = x
					    if(y<GminY):
						    GminY = y
						    #### Get the tip of the Green bone
						    GtipX = x
						    GtipY = y
					    if(x>GmaxX):
						    GmaxX = x
					    if(y>GmaxY):
						    GmaxY = y
				if(img_copy[y,x][0]==0 and img_copy[y,x][1]==0 and img_copy[y,x][2]>150): # blue color
					seeBlue = 1
					if(x<BminX):
						BminX = x
					if(y<BminY):
						BminY = y
					if(x>BmaxX):
						BmaxX = x
					if(y>BmaxY):
						BmaxY = y
		Rcenter = (-1,-1)
		RGcenter = (-1,-1)
		Bcenter = (-1,-1)
		if(seeRed):
			print("See Red")
			print("RminX,RmaxX,RminY,RMaxY = ",RminX,RmaxX,RminY,RmaxY)
			if((RmaxX > (RminX + 1)) and (RmaxY > (RminY + 1))):
				jetson.utils.cudaDrawRect(img_copy, (RminX,RminY,RmaxX,RmaxY), (128,0,0,100))
				Rx=int((RminX + RmaxX)/2)
				Ry=int((RminY + RmaxY)/2)
				Rcenter = (Rx,Ry)
				print("Rcenter = ",Rcenter)
				jetson.utils.cudaDrawCircle(img_copy, Rcenter, 5, (0,255,255,255))
				x1, y1 = int((RminX)),0
				x2, y2 = int((RminX)),img_copy.shape[0]
				starting_point = (x1,y1)
				ending_point = (x2,y2)
		if(seeGreen):
			print("See Green")
			print("GminX,GmaxX,GminY,GMaxY = ",GminX,GmaxX,GminY,GmaxY)
			if((GmaxX > (GminX + 1)) and (GmaxY > (GminY + 1))):
				jetson.utils.cudaDrawRect(img_copy, (GminX,GminY,GmaxX,GmaxY), (0,128,0,100))
				Gx=int((GminX + GmaxX)/2)
				Gy=int((GminY + GmaxY)/2)
				Gcenter = (Gx,Gy)
				print("Gcenter = ",Gcenter)
				jetson.utils.cudaDrawCircle(img_copy, Gcenter, 5, (255,0,255,255))
				x1, y1 = int((GminX + GmaxX)/2),(int((GminY + GmaxY)/2))
				x2, y2 = int((GminX + GmaxX)/2),(int((GminY + GmaxY)/2+100))
				Gstarting_point = (x1,y1)
				Gending_point = (x2,y2)  
		if(seeBlue):
			print("See Blue")
			print("BminX,BmaxX,BminY,BMaxY = ",BminX,BmaxX,BminY,BmaxY)
			if((BmaxX > (BminX + 1)) and (BmaxY > (BminY + 1))):
				jetson.utils.cudaDrawRect(img_copy, (BminX,BminY,BmaxX,BmaxY), (0,0,128,150))
				Bx=int((BminX + BmaxX)/2)
				By=int((BminY + BmaxY)/2)
				Bcenter = (Bx,By)
				print("Bcenter = ",Bcenter)
				jetson.utils.cudaDrawCircle(img_copy, Bcenter, 5, (255,255,0,255))
				x1, y1 = int((BminX + BmaxX)/2),0
				x2, y2 = int((BminX + BmaxX)/2),img_copy.shape[0]
				starting_point = (x1,y1)
				ending_point = (x2,y2)
				x1, y1 = int((BminX)),0
				x2, y2 = int((BminX)),img_copy.shape[0]
				starting_point = (x1,y1)
				ending_point = (x2,y2)

                ##########################
                #check centre point

		label = "Normal"
		RB=""
		BG=""
		if(Bcenter[0] == -1 or Gcenter[0] == -1):
		    print("Case 1: ")
		    label = "Abnormal:Case1"
		if((Bcenter[0] != -1) and (Rcenter[0] !=-1)):
			RB = Rcenter[0] - Bcenter[0]
			if((RB + BRtolerance)<0):
			    temp_str = "Case 2: " + "RB= " + str(RB)  + " BRtolerance= " + str(BRtolerance)
			    print(temp_str)
			    label = "Abnormal:Case2"
		if((Bcenter[0] != -1) and (Gcenter[0] !=-1)):
			BG = Bcenter[0] - GtipX
			if((BG + BGtolerance)<0):
			    temp_str = "Case 3: " + "BG= " + str(BG)  + " BGtolerance= " + str(BGtolerance)
			    print(temp_str)
			    label = "Abnormal:Case3"
		if(Bcenter[0] != -1):
			Bwidth = BmaxX - BminX
			Bheight = BmaxY - BminY
			if(Bwidth < Bmintolerance or (Bwidth > Bmaxtolerance and Bheight < BminHeight)):
			    temp_str = "Case 4: " + "Bwidth= " + str(Bwidth)   + "Bheight= " + str(Bheight) + " Bmintolerance= " + str(Bmintolerance) + " Bmaxtolerance= " + str(Bmaxtolerance)
			    print(temp_str)
			    label = "Abnormal:Case4"
		if(Bcenter[0] != -1 and Gcenter[0] != -1):
			if(Bcenter[1] > (GminY+Gmintolerance)):
			    temp_str = "Case 5: " + "Bcenter[1]= " + str(Bcenter[1])   + " GminY= " + str(GminY)  + " Gmintolerance= " + str(Gmintolerance)
			    print(temp_str)
			    label = "Abnormal:Case5"
			    
		text_out1 = label + "," + str(RB) + "," + str(BG)           
		text_out2 = str(RB)                   
		text_out3 = str(BG)                    
	jetson.utils.cudaResize(img_copy, buffers.mask)

        ##########################
	# composite the images
	if buffers.composite:
		jetson.utils.cudaOverlay(buffers.overlay, buffers.composite, 0, 0)
		jetson.utils.cudaOverlay(buffers.mask, buffers.composite, buffers.overlay.width, 0)
		jetson.utils.cudaOverlay(img_small, buffers.composite, buffers.overlay.width, buffers.mask.height)
		font.OverlayText(buffers.output, buffers.output.shape[1], buffers.output.shape[0], "{:s}".format(label), 5, 5, font.White, font.Gray10)
	output.Render(buffers.output)
	# update the title bar
	output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))
	# print out performance info
	jetson.utils.cudaDeviceSynchronize()
	net.PrintProfilerTimes()

    # compute segmentation class stats
	if opt.stats:
		buffers.ComputeStats()
    
	# exit on input/output EOS
	if not input.IsStreaming() or not output.IsStreaming():
		break
