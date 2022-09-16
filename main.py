import enum
import numpy as np
import cv2 as cv

# https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect
def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv.getPerspectiveTransform(rect, dst)
	warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

vid = cv.VideoCapture(0)
def reccuring(hir,index):
    tcCount = 0
    firstChild = hir[0,index,2]
    if firstChild > -1:
        tcCount = tcCount+1
        tcCount = tcCount + reccuring(hir, firstChild)
    return tcCount
def findFinder(hir):
    children = []
    for i in range(len(hir[0])):
        children.append(reccuring(hir,i))
    return children
def area(rect):
    v1 = rect[0][0]
    v2 = rect[1][0]
    v3 = rect[2][0]
    v4 = rect[3][0]
    return ((v1[0]*v2[1] - v1[1]*v2[0]) + (v2[0]*v3[1] - v2[1]*v3[0]) + (v3[0]*v4[1] - v3[1]*v4[0]) + (v4[0]*v1[1] - v4[1]*v1[0]))/2
    exit()
def convertToGoodVectors(arr):
    out = np.empty([len(arr),2])
    for i in range(len(arr)):
        out[i] = arr[i][0]
    return out

while(True):
    ret, frame = vid.read()
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    thresh_img = cv.threshold(gray_img, 0, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C + cv.THRESH_OTSU)[1]

    cnts,hir = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    out = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
    out = cv.drawContours(out,cnts,-1,(255,255,255),1)
    #if cv.waitKey(1) & 0xFF == ord(' '):
    """ for x in cnts:
        if len(x) > 4:
            print(x) """
    #print(hir[0])
    #print("hir",len(hir[0]))
    #print("cnts", len(cnts))
    rere = findFinder(hir)
    #print(rere)
    #print("rere",len(rere))
    for idx, x in enumerate(rere):
        if(x) < 2:
            continue
        #print(x)
        #empty = cv.drawContours(empty,[cnts[idx]],-1,(255,0,0),1)
        #if(len(cnts[idx]) == 4):
        if True:
            #print(cnts[idx])
            epsilon = 0.1*cv.arcLength(cnts[idx],True)
            approx = cv.approxPolyDP(cnts[idx],epsilon,True)
            if(len(approx) == 4):
                a = area(approx)
                cv.putText(out, str(a), approx[0][0], cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv.LINE_AA)
                out = cv.drawContours(out, approx, -1, (0,255,0),3)
                approx = convertToGoodVectors(approx)
                out = four_point_transform(frame,approx)
            else:
                print("shit")
                #break
    if cv.waitKey(1) & 0xff == ord("q"):
        break
    cv.imshow('feed',out)
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()