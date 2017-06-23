import numpy as np
import cv2


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


def get_squares(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thres = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    #thres = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    contours, hierarchy = cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
        if len(cnt) == 4 and cv2.contourArea(cnt) > 50 and cv2.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
            if max_cos < 0.1:
                squares.append(cnt)
    return squares

def get_perspective_img(img,aws):

    dim=700
    old_pts = np.float32([aws[0],aws[1],aws[2],aws[3]])
    
    (tl, tr, br, bl) = aws
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    new_pts = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")

    new_pts = np.float32([[0,0],[dim,0],[dim,dim],[0,dim]])
    maxWidth = dim
    maxHeight = dim
    
    M = cv2.getPerspectiveTransform(old_pts, new_pts)
    
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warp
def get_val_matrix(dd):
    val_mat=np.zeros((7,7))
    for i in range(0,7,1):
        st=i*100
        x=dd[st:st+100,:,:]
        for j in range(0,7,1):
            st=j*100
            y=x[:,st:st+100,:]
            y=np.ravel(y)
            val =np.argmax(np.bincount(y))
            val_mat[i][j]=val/255
    return val_mat[1:6].T[1:6]
def get_id(mat):
    idd=''
    fin=[]
    col1=mat[:,1]
    col2=mat[:,3]
    x=[col1,col2]
  
    for i in range(len(col1)):
        d=[col1[i],col2[i]]
        fin.append(d)
    fin=np.ravel(np.array(fin,dtype=int))
    for i in range(len(fin)):
        idd=idd+str(fin[i])
    return int(idd,2)
img=cv2.imread('aruco.jpg')
cv2.imshow('kld',img)
squares=get_squares(img);


for k in range(0,len(squares)):
    dd=get_perspective_img(img,squares[k])
    #cv2.drawContours( img, squares, j, (0, 255, 0), 3 )
    
    mat= get_val_matrix(dd)
    print mat
    print get_id(mat)
cv2.waitKey(0);
    
cv2.destroyAllWindows()