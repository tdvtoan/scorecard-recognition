from collections import Counter

import cv2
import numpy as np
from numpy.linalg import norm

SCALE_IMG_WIDTH = [700,650]

SCALE_IMG_HEIGHT = 450

bin_n = 16  # Number of bins
DIGITS_FN = 'train_data/digits.png'
SZ = 20  # size of each digit is SZ x SZ
CLASS_N = 10
DETECT_LIMIT = 0
is_capture = True
par = []
table = None

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class KNearest(StatModel):
    def __init__(self, k=3):
        self.k = k
        self.model = cv2.ml.KNearest_create()

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.findNearest(samples, self.k)
        return results.ravel()


class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1][0].ravel()


class ImageClass:
    def __init__(self):
        self.biggest = []
        # .maxArea is the area of this biggest rectangular found
        self.maxArea = []
        self.start = True
        self.sc_detect = 0

        self.store_biggest = None
        self.store_second = None

    def find_2_biggest_contour(self, img):
        img_copy = img
        # Convert to gray image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        image_area = gray.size  # this is area of the image

        # Threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        edged = cv2.Canny(gray, 10, 250)
        # cv2.imshow("Edged", edged)
        # cv2.waitKey(0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow("Closed", closed)
        # cv2.waitKey(0)
        # thresh = cv2.dilate(thresh, kernel, iterations=1)
        # Use open morphology to reduce noise
        # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # opening = cv2.dilate(thresh, kernel, iterations=2)
        # cv2.imshow('Opening', opening)
        # Find the biggest coutour
        img2, contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow('thresh', thresh)
        biggest = None
        max_contour = None
        max_area = 0
        #cv2.drawContours(img, contours, -1, (0, 255, 0), 1)


        for i in contours:
            area = cv2.contourArea(i)
            if area > image_area / 7:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
                    max_contour = i
        if max_contour is not None:
            self.store_biggest = biggest
            print('Biggest - ' + str(max_area))
            # cv2.drawContours(img, [max_contour], 0, (255, 0, 0), 2)
            # out = np.hstack([img])
            # cv2.imshow('Biggest Contour', out)
            # cv2.waitKey(0)
        second_max_area = 0
        second_contour = None
        second = None
        for i in contours:
            area = cv2.contourArea(i)
            if area > image_area / 7:
                peri = cv2.arcLength(i, True)
                #cv2.drawContours(img, [i], 0, (0, 0, 255), 2)
                # out = np.hstack([img])
                # cv2.imshow('Second Contour', out)
                # cv2.waitKey(0)
                #print(len(approx))
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > second_max_area and area < max_area and len(approx) == 4:
                    second = approx
                    second_max_area = area
                    second_contour = i

        if second_contour is not None:
            self.store_second = second
            print('Second- ' + str(second_max_area))
            # cv2.drawContours(img, [second_contour], 0, (0, 0, 255), 2)
            # out = np.hstack([img])
            # cv2.imshow('Second Contour', out)

        if max_area > 0 and biggest is not None and second_max_area > 0 and second is not None:
            self.sc_detect += 1
            # cv2.drawContours(img, [max_contour], 0, (255, 0, 0), 2)
            # cv2.drawContours(img, [second_contour], 0, (255, 0, 0), 2)
            # out = np.hstack([img])
            # cv2.imshow('All Contour', out)
        if self.sc_detect > DETECT_LIMIT:
            (bx1, by1, bw1, bh1) = cv2.boundingRect(max_contour)
            # cv2.imshow('Extract Img', img[by1:by1 + bh1, bx1:bx1 + bw1])
            (bx2, by2, bw2, bh2) = cv2.boundingRect(second_contour)
            # cv2.imshow('Extract Second Img', img[by2:by2 + bh2, bx2:bx2 + bw2])
            if bx1 > bx2:
                self.biggest = [self.store_second, self.store_biggest]
            else:
                self.biggest = [self.store_biggest, self.store_second]
                # cv2.drawContours(img, [second_contour], 0, (0, 255, 0), 1)
                # cv2.drawContours(img, [max_contour], 0, (255, 0, 0), 1)
                # out = np.hstack([img])
                # cv2.imshow('Detect', out)


class ScoreCardReg:
    def __init__(self):
        digits, labels = self.load_digits(DIGITS_FN)
        # shuffle digits
        rand = np.random.RandomState(321)
        shuffle = rand.permutation(len(digits))
        digits, labels = digits[shuffle], labels[shuffle]

        digits2 = list(map(self.deskew, digits))
        samples = self.preprocess_hog(digits2)

        train_n = int(0.9 * len(samples))
        samples_train, samples_test = np.split(samples, [train_n])
        labels_train, labels_test = np.split(labels, [train_n])

        model_svm = SVM(C=2.67, gamma=5.383)
        model_svm.train(samples_train, labels_train)

        model_knn = KNearest(k=4)
        model_knn.train(samples_train, labels_train)

        self.model = model_knn
        print('Finish Init KNN+SVM...')

    def load_digits(self, fn):
        digits_img = cv2.imread(fn, 0)
        digits = self.split2d(digits_img, (SZ, SZ))
        labels = np.repeat(np.arange(CLASS_N), len(digits) / CLASS_N)
        return digits, labels

    def split2d(self, img, cell_size, flatten=True):
        h, w = img.shape[:2]
        sx, sy = cell_size
        cells = [np.hsplit(row, w // sx) for row in np.vsplit(img, h // sy)]
        cells = np.array(cells)
        if flatten:
            cells = cells.reshape(-1, sy, sx)
        return cells

    def deskew(self, img):
        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            return img.copy()
        skew = m['mu11'] / m['mu02']
        M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        return img

    def preprocess_simple(self, digits):
        return np.float32(digits).reshape(-1, SZ * SZ) / 255.0

    def preprocess_hog(self, digits):
        samples = []
        for img in digits:
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
            mag, ang = cv2.cartToPolar(gx, gy)
            bin_n = 16
            bin = np.int32(bin_n * ang / (2 * np.pi))
            bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
            mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
            hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
            hist = np.hstack(hists)

            # transform to Hellinger kernel
            eps = 1e-7
            hist /= hist.sum() + eps
            hist = np.sqrt(hist)
            hist /= norm(hist) + eps

            samples.append(hist)
        return np.float32(samples)

    def hog(self, img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)

        # quantizing binvalues in (0...16)
        bins = np.int32(bin_n * ang / (2 * np.pi))

        # Divide to 4 sub-squares
        bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps
        return hist

    def remove_line(self, img):
        warpg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # kept a gray-scale copy of warp for further use

        mask = np.zeros((warpg.shape),np.uint8)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        close = cv2.morphologyEx(warpg,cv2.MORPH_CLOSE,kernel1)
        div = np.float32(warpg)/(close)
        warpg = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
        #warpg = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)

        # --------------- Now take eac   h element for inspection --------------------------
        smooth = cv2.GaussianBlur(warpg, (3,3), 3)
        thresh = cv2.adaptiveThreshold(smooth.copy(), 255, 0, 1, 11, 3)
        cv2.imshow('edge', thresh)
        cv2.waitKey(0)
        #kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))
        #

        # ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        close = cv2.adaptiveThreshold(smooth.copy(), 0, 0, 1, 11, 3)
        black_img = close.copy()
        edges = cv2.Canny(warpg,100,150,apertureSize = 3)
        #edges = cv2.dilate(edges, kernel, iterations=1)
        # dx = cv2.Sobel(edges,cv2.CV_16S,1,0)
        # dx = cv2.convertScaleAbs(dx)
        #
        # cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
        edges = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,kernel,iterations = 1)
        # cv2.imshow('edge', edges)
        # cv2.waitKey(0)

        # dy = cv2.Sobel(edges,cv2.CV_16S,0,1)
        # dy = cv2.convertScaleAbs(dy)
        # cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
        # cv2.imshow('edge', dy)
        # cv2.waitKey(0)

        minLineLength = 80
        maxLineGap = 10

        lines = cv2.HoughLinesP(edges,1,np.pi/180,80,minLineLength=minLineLength,maxLineGap=maxLineGap)

        for l in lines:
            for x1,y1,x2,y2 in l:
                cv2.line(close, (x1,y1), (x2,y2), (255,255,255), 1, 8)
        close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,kernel,iterations=2)
        close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernel,iterations=1)
        # cv2.imshow('hough', close)
        # cv2.waitKey(0)

        #close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)

        # img2, contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # for cnt in contour:
        #     x,y,w,h = cv2.boundingRect(cnt)
        #     if h/w > 5:
        #         cv2.drawContours(close,[cnt],0,255,-1)
        #     else:
        #         cv2.drawContours(close,[cnt],0,0,-1)
        #
        # close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations=1)
        # ver_img = close.copy()
        #
        # kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
        # dy = cv2.Sobel(smooth,cv2.CV_16S,0,1)
        # dy = cv2.convertScaleAbs(dy)
        # cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
        #
        # ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # cv2.imshow('thresh', close)
        # cv2.waitKey(0)
        # #close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely)
        #
        # img2, contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # for cnt in contour:
        #     x,y,w,h = cv2.boundingRect(cnt)
        #     if w/h > 5:
        #         cv2.drawContours(close,[cnt],0, (255, 0, 0), 1)
        #     else:
        #         cv2.drawContours(close,[cnt],0,0,-1)
        #
        # close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 1)
        # hor_img = close.copy()


        #cv2.imshow('Thresh', thresh)
        # gb = cv2.bitwise_not(thresh)
        # cv2.imshow('Bitwise', gb)
        #img_cp = img.copy()
        #v = img.mean()
        # apply automatic Canny edge detection using the computed median
        # lower = int(max(0, (1.0 - 0.33) * v))
        # upper = int(min(255, (1.0 + 0.33) * v))
        # edges = cv2.Canny(warpg,5,150,apertureSize=3)
        # cv2.imshow('canny', edges)
        # minLineLength = 100
        # maxLineGap = 10
        # lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
        # for x1,y1,x2,y2 in lines[0]:
        #     cv2.line(img_cp,(x1,y1),(x2,y2),(0,255,0),2)
        #
        # cv2.imshow('houghlines',img_cp)

        # ---- Remove lines ----
        # height, width, channels = img.shape
        #
        # # Specify size on horizontal axis
        # horizontalsize = int(height / 5)
        # verticalsize = int(width / 20)
        #
        # # Create structure element for extracting horizontal lines through morphology operation
        # hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
        # ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        #
        # # Apply morphology operations
        # ver_opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, ver_kernel, iterations=1)
        # hor_opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, hor_kernel, iterations=1)
        #
        # # Eliminate vertical and horizontal lines
        substract_img = edges - close
        #
        # substract_img = cv2.morphologyEx(substract_img, cv2.MORPH_OPEN, kernel)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        # #substract_img = cv2.dilate(substract_img, kernel, iterations=1)
        #substract_img = cv2.morphologyEx(substract_img, cv2.MORPH_CLOSE, kernel)

        # cv2.imshow('ver', ver_img)
        # cv2.waitKey(0)
        # cv2.imshow('hor', hor_img)
        # cv2.waitKey(0)



        ver_black = black_img.copy()
        img2, contour, hier = cv2.findContours(substract_img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            x,y,w,h = cv2.boundingRect(cnt)
            if h/w > 10:
                cv2.drawContours(ver_black,[cnt],0,(255, 255, 255), 1)
            else:
                cv2.drawContours(ver_black,[cnt],0,(0, 0, 0), 1)


        ver_black = cv2.morphologyEx(ver_black,cv2.MORPH_DILATE,kernel,iterations = 1)
        ver_img = ver_black.copy()

        hor_black = black_img.copy()
        img2, contour, hier = cv2.findContours(substract_img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            x,y,w,h = cv2.boundingRect(cnt)
            if w/h > 10:
                cv2.drawContours(hor_black,[cnt],0,(255, 255, 255), 1)
            else:
                cv2.drawContours(hor_black,[cnt],0,(0, 0, 0), 1)

        hor_black = cv2.morphologyEx(hor_black,cv2.MORPH_DILATE,kernel,iterations=1)
        hor_img = hor_black.copy()

        substract_img = substract_img - ver_img - hor_img

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        #substract_img = cv2.Canny(substract_img,100,150,apertureSize=3)
        substract_img = cv2.morphologyEx(substract_img,cv2.MORPH_CLOSE,kernel,iterations=1)
        # h, w = substract_img.shape[:2]
        # mask = np.zeros((h+2, w+2), np.uint8)
        # cv2.floodFill(substract_img, mask, (0,0), 255)
        ret, substract_img = cv2.threshold(substract_img, 127, 255, cv2.THRESH_BINARY)
        
        return substract_img

    def warp_image(self, approx, img, index):
        h = np.array(
            [[0, 0], [SCALE_IMG_WIDTH[index] - 1, 0], [SCALE_IMG_WIDTH[index] - 1, SCALE_IMG_HEIGHT - 1], [0, SCALE_IMG_HEIGHT - 1]],
            np.float32)  # this is corners of new square image taken in CW order

        approx = self.rectify(approx)  # we put the corners of biggest square in CW order to match with h

        retval = cv2.getPerspectiveTransform(approx, h)  # apply perspective transformation
        warp = cv2.warpPerspective(img, retval,
                                   (SCALE_IMG_WIDTH[index], SCALE_IMG_HEIGHT))  # Now we get perfect square with size 450x450
        return warp

    # ---------------- Function to put vertices in clockwise order ----------
    def rectify(self, h):
        ''' this function put vertices of square we got, in clockwise order '''
        h = h.reshape((4, 2))
        hnew = np.zeros((4, 2), dtype=np.float32)

        add = h.sum(1)
        hnew[0] = h[np.argmin(add)]
        hnew[2] = h[np.argmax(add)]

        diff = np.diff(h, axis=1)
        hnew[1] = h[np.argmin(diff)]
        hnew[3] = h[np.argmax(diff)]

        return hnew

    # ------- Find most popular background -----------------
    def isMostWhiteBg(self, img):
        colors_count = {}
        channel_b, channel_g, channel_r = cv2.split(
            img)  # Splits image Mat into 3 color channels in individual 2D arrays

        channel_b = channel_b.flatten()
        channel_g = channel_g.flatten()
        channel_r = channel_r.flatten()
        for i in range(len(channel_b)):
            RGB = (channel_r[i], channel_g[i], channel_b[i])
            if RGB in colors_count:
                colors_count[RGB] += 1
            else:
                colors_count[RGB] = 1
        # print(colors_count)
        number_counter = Counter(colors_count).most_common(20)
        red = 0
        green = 0
        blue = 0
        sample = 20
        for top in range(0, sample):
            red += number_counter[top][0][0]
            green += number_counter[top][0][1]
            blue += number_counter[top][0][2]

        average_red = red / sample
        average_green = green / sample
        average_blue = blue / sample
        # print(str(average_red) + ',' + str(average_green) + ',' + str(average_blue))
        if average_red >= 128 and average_green >= 128 and average_blue >= 128:
            return True
        else:
            return False

    def readScoreCard(self, img, biggest, model):
        sc = [np.zeros((9, 14), np.uint8), np.zeros((9, 15), np.uint8)]
        read_index = [(4, 13), (0, 9)]
        kernel = np.matrix('0,1,0;1,1,1;0,1,0', np.uint8)
        # ------- Remove all lines  -------
        # warp = cv2.imread('warp.png')
        par = []
        memberId = -1
        result = {}
        # cv2.imshow('Origin', img)
        store_memberID = []
        for index, approx in enumerate(biggest):
            print("Solving index - " + str(index))
            warp = self.warp_image(approx, img,index)
            warpg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            smooth = cv2.GaussianBlur(warpg, (3,3), 3)
            thresh = cv2.adaptiveThreshold(smooth.copy(), 255, 0, 1, 11, 3)
            # cv2.imshow('Warp-' + str(index), warp)
            non_line_img = self.remove_line(warp)
            cv2.imshow('Remove-Line-' + str(index), non_line_img)

            cv2.waitKey(0)
            mor_img_copy = non_line_img.copy()
            # find contours
            img2, contours, hierarchy = cv2.findContours(mor_img_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            warp_copy = warp.copy()
            diffWhite = 0
            # cv2.drawContours(warp_copy, contours, -1, (255, 255, 0), 1)
            # out = np.hstack([warp_copy])
            # cv2.imshow('All Contours', out)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # cv2.drawContours(warp_copy, [cnt], 0, (255, 255, 0), 1)
                # out = np.hstack([warp_copy])
                # cv2.imshow('Ouput', out)
                # cv2.waitKey(0)
                #print('area' + str(area))
                if 80 < area < 1200:
                    (bx, by, bw, bh) = cv2.boundingRect(cnt)
                    try:
                        # out = np.hstack([warp[by - 10:by + bh + 10, bx - 10:bx + bw + 10]])
                        # cv2.imshow('Ouput', out)
                        # cv2.waitKey(0)
                        if self.isMostWhiteBg(warp[by - 10:by + bh + 10, bx - 10:bx + bw + 10]):
                            roi = non_line_img[by:by + bh, bx:bx + bw].astype(np.uint8)
                            cv2.imshow('ROi', roi)
                            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                            roi = cv2.erode(roi, kernel, iterations=1)

                            small_roi = cv2.resize(roi, (SZ, SZ))

                            #cv2.waitKey(0)
                            hog_feature = self.hog(small_roi)
                            # hog_feature = preprocess_hog(roi)
                            feature = np.float32(hog_feature).reshape(-1, bin_n * 4)
                            # feature = small_roi.reshape((1, SZ * SZ)).astype(np.float32)
                            results = model.predict(feature)
                            #if index == 0:
                            cv2.drawContours(warp_copy, [cnt], 0, (255, 0, 0), 1)
                            out = np.hstack([warp_copy])
                            cv2.imshow('Output', out)
                            print('knn - ' + str(results))


                            gridy, gridx = (bx + bw / 2) / 50, (
                                by + bh / 2) / 50  # gridx and gridy are indices of row and column in sudo
                            print gridx,gridy

                            integer = int(results.ravel()[0])
                            if index == 0 and gridy < 4:
                                valid = True
                                for item in store_memberID:
                                    if abs(item['x'] - bx) <= 10 and item['gridx'] == gridx:
                                        valid = False
                                        break
                                if valid:
                                    store_memberID.append({
                                        'number':integer,
                                        'x':bx,
                                        'gridx':gridx
                                    })
                            sc[index].itemset((gridx, gridy), integer)
                            print sc[index]
                            cv2.waitKey(0)
                        else:
                            diffWhite += 1
                            if diffWhite == 5:
                                break
                    except Exception as e:
                        print(e)
                        pass
            #non_zero_row = sc[index][~np.all(sc[index] == 0, axis=1)]
            non_zero_row = sc[index]

            map = []
            for i, r in enumerate(non_zero_row):
                map.append({'count':np.count_nonzero(r), 'index':int(i)})

            sorted_map = sorted(map, key=lambda m: (m['count']), reverse=True)

            max_i = sorted_map[0]['index']
            result.update({index: non_zero_row})
            for i in range(read_index[index][0], read_index[index][1]):
                par.append(non_zero_row.item(max_i, i))
            if index == 0:
                print store_memberID
                par_row = [x for x in store_memberID if x['gridx'] == max_i]
                par_row = sorted(par_row, key=lambda m: (m['x']),reverse=True)
                memberId = 0
                for i, item in enumerate(par_row):
                    memberId += item['number']*10**i
                print memberId
                # for i in range(0, read_index[0][0]):
                #     if non_zero_row.item(max_i, i) != 0:
                #         memberId = non_zero_row.item(max_i, i)

            print("----------- Result ----------------")
            print(result)

        return par, memberId

    def process(self, img):
        pre_img = ImageClass()
        pre_img.find_2_biggest_contour(img)
        par, memberId = self.readScoreCard(img, pre_img.biggest, self.model)
        return par, memberId
