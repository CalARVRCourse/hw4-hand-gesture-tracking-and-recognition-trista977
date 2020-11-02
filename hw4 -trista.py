from __future__ import print_function
import cv2
import numpy as np
import pyautogui 

max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
trackbar_blur = 'Blur kernel size'
window_name = 'Threshold Demo'
isColor = False
font = cv2.FONT_HERSHEY_SIMPLEX 

def nothing(x):
    pass
  



cam = cv2.VideoCapture(0)
cv2.namedWindow(window_name)
cv2.createTrackbar(trackbar_type, window_name , 3, max_type, nothing)
# Create Trackbar to choose Threshold value
cv2.createTrackbar(trackbar_value, window_name , 0, max_value, nothing)
# Call the function to initialize
cv2.createTrackbar(trackbar_blur, window_name , 1, 20, nothing)
# create switch for ON/OFF functionality
color_switch = 'Color'
cv2.createTrackbar(color_switch, window_name,0,1,nothing)
cv2.createTrackbar('Contours', window_name,0,1,nothing)
while True:
    ret, frame1 = cam.read()
    frame = cropImg = frame1[350:850, 350:850]
    if not ret:
        break
    
    #0: Binary
    #1: Binary Inverted
    #2: Threshold Truncated
    #3: Threshold to Zero
    #4: Threshold to Zero Inverted
    threshold_type = cv2.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv2.getTrackbarPos(trackbar_value, window_name)
    blur_value = cv2.getTrackbarPos(trackbar_blur, window_name)
    blur_value = blur_value+ (  blur_value%2==0)
    isColor = (cv2.getTrackbarPos(color_switch, window_name) == 1)
    findContours = (cv2.getTrackbarPos('Contours', window_name) == 1)
    
    #convert to grayscale
    if isColor == False:
        src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, dst = cv2.threshold(src_gray, threshold_value, max_binary_value, threshold_type )
        blur = cv2.GaussianBlur(dst,(blur_value,blur_value),0)
        if findContours:
            _, contours, hierarchy = cv2.findContours( blur, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE )
            blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)  #add this line
            output = cv2.drawContours(blur, contours, -1, (0, 255, 0), 1)
            print(str(len(contours))+"\n")
        else:
            output = blur
        
        
    else:
        _, dst = cv2.threshold(frame, threshold_value, max_binary_value, threshold_type )
        blur = cv2.GaussianBlur(dst,(blur_value,blur_value),0)
        output = blur
    
    
    
    #part 1
    lower_HSV = np.array([0, 40, 0], dtype = "uint8")  
    upper_HSV = np.array([17, 170, 255], dtype = "uint8")  
  
    convertedHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
    skinMaskHSV = cv2.inRange(convertedHSV, lower_HSV, upper_HSV)  
  
  
    lower_YCrCb = np.array((0, 138, 67), dtype = "uint8")  
    upper_YCrCb = np.array((255, 173, 133), dtype = "uint8")  
      
    convertedYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)  
    skinMaskYCrCb = cv2.inRange(convertedYCrCb, lower_YCrCb, upper_YCrCb)  
      
    skinMask = cv2.add(skinMaskHSV,skinMaskYCrCb)  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)  
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)  
  
    # blur the mask to help remove noise, then apply the  
    # mask to the frame  
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0) 
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    skin = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    #output for part1
    cv2.imshow("part1", skin)

    
    #part2
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
    ret, thresh = cv2.threshold(gray, 0, max_binary_value, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU )  
    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(thresh,ltype=cv2.CV_16U)  
    markers = np.array(markers, dtype=np.uint8)  
    label_hue = np.uint8(179*markers/np.max(markers))  
    blank_ch = 255*np.ones_like(label_hue)  
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img,cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0  
    cv2.imshow("part2", labeled_img)
    
    statsSortedByArea = stats[np.argsort(stats[:, 4])]  
    roi = statsSortedByArea[-3][0:4]  
    x, y, w, h = roi  
    subImg = labeled_img[y:y+h, x:x+w]  
    if (ret>2):  
        try:  
            roi = statsSortedByArea[-3][0:4]  
            x, y, w, h = roi  
            subImg = labeled_img[y:y+h, x:x+w]  
            subImg = cv2.cvtColor(subImg, cv2.COLOR_BGR2GRAY);  
            subImg=cv2.medianBlur(subImg,5)
            _, contours, _ = cv2.findContours(subImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
            maxCntLength = 0  
            for i in range(0,len(contours)):  
                cntLength = len(contours[i])  
                if(cntLength>maxCntLength):  
                    cnt = contours[i]  
                    maxCntLength = cntLength  
            if(maxCntLength>=5):  
                ellipseParam = cv2.fitEllipse(cnt)  
                subImg = cv2.cvtColor(subImg, cv2.COLOR_GRAY2RGB);  
                subImg = cv2.ellipse(subImg,ellipseParam,(0,255,0),2)  
              
            subImg = cv2.resize(subImg, (0,0), fx=3, fy=3)  
            (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)  
            print ('------------------------')
            print ("central points: (%f, %f)" %(x,y))
            print ("Major axis = %f, minor axis = %f" %(MA, ma))
            print ("rotation angle is %f degree" %(angle))
            #output for part2
            cv2.imshow("ROI "+str(2), subImg)  
            cv2.waitKey(1)  
        except:  
            print("No hand found")  

    
    
    #part3    
    #redue the noise of thresholdedHandImage
    thresholdedHandImage = frame
    fgbg = cv2.createBackgroundSubtractorMOG2()  
    fgmask = fgbg.apply(thresholdedHandImage)
    kernel2 = np.ones((5, 5), np.uint8)
    fgmask = cv2.erode(fgmask, kernel2, iterations=1)  # Erosion
    fgmask = cv2.dilate(fgmask, kernel2, iterations=1)  # Dilation
    res2 = cv2.bitwise_and(thresholdedHandImage, thresholdedHandImage, mask=fgmask)
    ycrcb = cv2.cvtColor(res2, cv2.COLOR_BGR2YCrCb)  # divide in to YUV image, then get CR number
    (_, cr, _) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)  # GaussianBlur
    _, skin2 = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  
    gesture_roi = skin2

    _, contours, _ = cv2.findContours(gesture_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)       
    contours=sorted(contours,key=cv2.contourArea,reverse=True)       
    if len(contours)>1:  
        largestContour = contours[0]  
        hull = cv2.convexHull(largestContour, returnPoints = False)     
        for cnt in contours[:1]:  
            fingerCount = 0
            defects = cv2.convexityDefects(cnt,hull)  
            if(not isinstance(defects,type(None))):  
                for i in range(defects.shape[0]):  
                    s,e,f,d = defects[i,0]  
                    start = tuple(cnt[s][0])  
                    end = tuple(cnt[e][0])  
                    far = tuple(cnt[f][0])  
                    
                    
# =============================================================================
#                     cv2.line(thresholdedHandImage, start, end, [255, 255, 0], 2)
#                     cv2.circle(thresholdedHandImage, far, 5, [0, 0, 255], -1) 
# =============================================================================
# =============================================================================
#                     M = cv2.moments(largestContour)
#                     cX = int(M[ "m10" ] / M[ "m00" ])
#                     cY = int(M[ "m01" ] / M[ "m00" ])
#                     print ('------------------------')
#                     #print("the area of the contour is %f" %(M))
#                     print(cX)
#                     print(cY)
# =============================================================================

                    #part3b 
                    c_squared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
                    a_squared = (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2
                    b_squared = (end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2
                    angle = np.arccos((a_squared + b_squared - c_squared ) / (2 * np.sqrt(a_squared * b_squared)))
                    
                    if angle <= np.pi / 3:
                       fingerCount += 1
                       cv2.line(thresholdedHandImage, start, end, [255, 255, 0], 2)
                       cv2.circle(thresholdedHandImage, far, 4, [0, 0, 255], -1) 
                    
                    
                    
# =============================================================================
#             if  fingerCount == 0:
#                 sm = 1
# =============================================================================
            if  fingerCount != 0:
                        fingerCount += 1
            cv2.putText(thresholdedHandImage, "finger count = %d" %(fingerCount), \
                                (50,350), font, 1, (255, 255, 0), 2)
            
                
            #part4  simple gestue, use the vertical four gesture to trigger the screenshot
            (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)  
            for angle in (113,118):
                if fingerCount == 4:
                    im1 = pyautogui.screenshot('my_screenshot1.png')
                    cv2.putText(thresholdedHandImage, "Picture captured", \
                                (50,300), font, 0.8, (255, 255, 0), 2)    
            
            
            #part4  complex gestue, i start from the use "yeah" gesture. \
            #    When I rotate my fingers left with two fingers attached, then turn the volume up
            #    When I rotate my fingers right with two fingers attached, then turn the volume down
            
            
             
            (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)  
            preangle = 0
            preMA = 0
            prema = 0
            multiFrameAngle = []
            frameThres = 20

            tAngle = 0.2
            if (len(multiFrameAngle) < frameThres):
                if (fingerCount ==2):
                    multiFrameAngle.append(angle)
            elif (len(multiFrameAngle) == frameThres):
                if (fingerCount ==2):
                    preangle = angle
                    preMA = MA
                    prema = ma
                    averageAngle =np.average(multiFrameAngle)
           
                if preangle in (14,17) and preMA/prema in(0.7,0.75):
                    if angle in (80,85) and MA/ma in(0.55,0.6):
                        pyautogui.press('F12')  
                        cv2.putText(thresholdedHandImage, "Volume Up" , 
                                        (50,300), font, 0.8, (255, 255, 0), 2)
                    
                    elif angle in (25,35) and MA/ma in(0.3,0.35):
                        pyautogui.hotkey('F11')  
                        cv2.putText(thresholdedHandImage, "Volume Down" , 
                                        (50,300), font, 0.8, (255, 255, 0), 2)
        
        
            #output for 3a
            cv2.imshow("part3a", thresholdedHandImage)
 


    cv2.imshow(window_name, output)
    k = cv2.waitKey(1) #k is the key pressed
    if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
        #exit
        cv2.destroyAllWindows()
        cam.release()
        break


