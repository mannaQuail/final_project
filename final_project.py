import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random

def makeGaussianKernel(size, sigma):
	center = (size-1)/2
	kernel = np.zeros((size, size))

	for x in range(size):
		for y in range(size):
			length = (x-center)**2 + (y-center)**2
			kernel[x][y] = np.exp(-length/(2*sigma**2))

	kernel /= kernel.sum()

	return kernel



def convolution2D(img, kernel, padding="reflection"):
	height = img.shape[0]
	width = img.shape[1]
	kernel_size = kernel.shape[0]
	tmp_img = np.zeros((height+kernel_size-1,width+kernel_size-1))
	result_img = np.zeros((height,width))
	half_kernel_size = int((kernel_size-1)/2)

	if padding=="reflection":
		tmp_img[half_kernel_size:half_kernel_size+height, half_kernel_size:half_kernel_size+width] = img[:, :]
		tmp_img[:half_kernel_size, half_kernel_size:half_kernel_size+width] = img[half_kernel_size:0:-1, :] #top
		tmp_img[half_kernel_size+height:, half_kernel_size:half_kernel_size+width] = img[height-2:height-half_kernel_size-2:-1, :] #bottom
		tmp_img[half_kernel_size:half_kernel_size+height, :half_kernel_size] = img[:, half_kernel_size:0:-1] #left
		tmp_img[half_kernel_size:half_kernel_size+height, half_kernel_size+width:] = img[:, width-2:width-half_kernel_size-2:-1] #right
		tmp_img[:half_kernel_size, :half_kernel_size] = img[half_kernel_size:0:-1, half_kernel_size:0:-1] #top left
		tmp_img[:half_kernel_size, half_kernel_size+width:] = img[half_kernel_size:0:-1, width-2:width-half_kernel_size-2:-1] #top right
		tmp_img[half_kernel_size+height:, :half_kernel_size] = img[height-2:height-half_kernel_size-2:-1, half_kernel_size:0:-1] #bottom left
		tmp_img[half_kernel_size+height:, half_kernel_size+width:] = img[height-2:height-half_kernel_size-2:-1, width-2:width-half_kernel_size-2:-1] #bottom right

	elif padding=="zero":
		pass
	else:
		raise ValueError("unexpected padding value")

	for x in range(height):
		for y in range(width):
			convol_sum = 0
			for i in range(kernel_size):
				for j in range(kernel_size):
					convol_sum += tmp_img[x+i,y+j]*kernel[i,j]
			result_img[x,y] = convol_sum

	return result_img



def harrisCornerDetection(img, box_rad, k):
	height = img.shape[0]
	width = img.shape[1]
	diff_x_kernel = np.array(
		[[-1, -2, -1],
		[0, 0, 0],
		[1, 2, 1]]
		)
	diff_y_kernel = np.array(
		[[-1, 0, 1],
		[-2, 0, 2],
		[-1, 0, 1]]
		)
	diff_x = convolution2D(img, diff_x_kernel)
	diff_y = convolution2D(img, diff_y_kernel)

	mask = np.full((height,width), 0)

	for x in range(height-box_rad*2):
		for y in range(width-box_rad*2):
			Ix2 = 0
			Iy2 = 0
			Ixy = 0

			for i in range(box_rad*2+1):
				for j in range(box_rad*2+1):
					Ix2 += diff_x[x+i,y+j]**2
					Iy2 += diff_y[x+i,y+j]**2
					Ixy += diff_x[x+i,y+j]*diff_y[x+i,y+j]

			det = Ix2*Iy2 - Ixy**2
			trace = Ix2 + Iy2
			R = det - k*(trace**2)

			mask[x+box_rad,y+box_rad] = R/1000

	return mask

def nonMaximumSuppression(image, window_rad):
	img = image.copy()
	height = img.shape[0]
	width = img.shape[1]

	for x in range(height):
		for y in range(width):
			if img[x,y] != 0:
				max_R = 0

				for i in range(2*window_rad+1):
					for j in range(2*window_rad+1):
						if x-window_rad+i>=0 and x-window_rad+i<height and y-window_rad+j>=0 and y-window_rad+j<width:
							if img[x-window_rad+i,y-window_rad+j] > max_R:
								max_R = img[x-window_rad+i,y-window_rad+j]


				for i in range(2*window_rad+1):
					for j in range(2*window_rad+1):
						if x-window_rad+i>=0 and x-window_rad+i<height and y-window_rad+j>=0 and y-window_rad+j<width:
							if img[x-window_rad+i,y-window_rad+j] < max_R:
								img[x-window_rad+i,y-window_rad+j] = 0

	result = []
	for x in range(height):
		for y in range(width):
			if img[x,y]!=0:
				result.append((x,y))


	return result



def findCorners(image):
	print("making gaussian kernel...")
	kernel = makeGaussianKernel(7, 7)

	print("executing convolution...")
	blured_image = convolution2D(image, kernel)

	print("executing harris corner detection...")
	mask = harrisCornerDetection(blured_image, 3, 0.04)

	mask_max = 0
	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			if mask[x,y] > mask_max:
				mask_max = mask[x,y]



	image_tmp_1 = image.copy()
	for x in range(image_tmp_1.shape[0]):
		for y in range(image_tmp_1.shape[1]):
			if mask[x,y] < 0.01*mask_max:
				mask[x,y] = 0

	cv2.imwrite("./harris.png", image_tmp_1)

	print("executing NMS...")
	corners = nonMaximumSuppression(mask, 30)

	image_tmp_2 = image.copy()
	for i in corners:
		image_tmp_2[i[0],i[1]] = 255

	cv2.imwrite("./NMS.png", image_tmp_2)

	return corners

def findCorrespondence(img1, corners1, img2, corners2, window_rad, threshold):
	correspond_min_point = []
	for cord1 in corners1:
		x1 = cord1[0]
		y1 = cord1[1]
		window1 = img1[x1-window_rad:x1+window_rad+1,y1-window_rad:y1+window_rad+1]

		correspond_min = float('inf')
		flag = False

		for cord2 in corners2:
			x2 = cord2[0]
			y2 = cord2[1]
			window2 = img2[x2-window_rad:x2+window_rad+1,y2-window_rad:y2+window_rad+1]

			correspond = 0
			if window1.shape[0]==2*window_rad+1 and window1.shape[1]==2*window_rad+1 and window2.shape[0]==2*window_rad+1 and window2.shape[1]==2*window_rad+1:
				for i in range(2*window_rad+1):
					for j in range(2*window_rad+1):
						correspond += (int(window1[i,j]) - int(window2[i,j]))**2

				if correspond < correspond_min:
					correspond_min = correspond
					correspond_point = [x2,y2]
					flag = True

		if flag:
			point_tmp = ((x1,y1),correspond_point)
			correspond_min_point.append(point_tmp)

	return correspond_min_point

import random
import numpy as np

def ransac(correspond, point_num, num_repeat, threshold, img1_width):
    best_inlier = []
    epsilon = 1e-6

    for repeat in range(num_repeat):
        # 1. 랜덤 샘플링
        sampled_points = random.sample(correspond, point_num)

        inclinations = []
        for (y1, x1), (y2, x2) in sampled_points:
            x2 += img1_width
            if abs(x1 - x2) > epsilon:
                inclinations.append(np.arctan((y1 - y2) / float(x1 - x2)))

        if not inclinations: 
            continue

        inclination = np.median(inclinations)

        inliers = []
        for (y1, x1), (y2, x2) in correspond:
            x2 += img1_width
            if abs(x1 - x2) < epsilon:  # 오버플로우 방지
                continue
            tmp_inclination = np.arctan((y1 - y2) / float(x1 - x2))
            if abs(tmp_inclination - inclination) < threshold:
                inliers.append(((y1, x1), (y2, x2-img1_width)))

        if len(inliers) > len(best_inlier):
            best_inlier = inliers

    return best_inlier

def findHomography(correspond, num_repeat):
	error_min = float('inf')
	result_H = np.zeros((3,3))
	for j in range(num_repeat):
		mat = np.zeros((8,9))
		tmp = random.sample(correspond, 4)
		for i in range(4):
			(x1, y1), (x2, y2) = tmp[i]
			row1 = [x2, y2, 1, 0, 0, 0, -x2*x1, -y2*x1, -x1]
			row2 = [0, 0, 0, x2, y2, 1, -x2*y1, -y2*y1, -y1]
			mat[i*2,:] = row1[:]
			mat[i*2+1,:] = row2[:]

		U, s, VT = np.linalg.svd(mat)

		H = VT[-1].reshape(3, 3)

		H = H / H[2, 2]

		error = 0

		for cor in correspond:
			(x1, y1), (x2, y2) = cor
			x_pred = H[0][0]*x2 + H[0][1]*y2 + H[0][2]
			y_pred = H[1][0]*x2 + H[1][1]*y2 + H[1][2]
			z = H[2][0]*x1 + H[2][1]*y1 + H[2][2]

			x_pred /= z
			y_pred /= z

			error += (x_pred-x1)**2 + (y_pred-y1)**2

		if error < error_min:
			error_min = error
			result_H = H


	return result_H

def stiching(img_src, img_dst, H, size):
	result = np.zeros(size)
	result = result.astype(np.uint8)
	x = 0
	y = 0
	for i in range(img_src.shape[0]):
		for j in range(img_src.shape[1]):
			x = H[0][0]*i + H[0][1]*j + H[0][2]
			y = H[1][0]*i + H[1][1]*j + H[1][2]
			z = H[2][0]*i + H[2][1]*j + H[2][2]
			x /= z
			y /= z
			x = round(x) + size[0]//2-img_dst.shape[0]//2
			y = round(y) + size[1]//2-img_dst.shape[1]//2
			if(x>=0 and x<size[0] and y>=0 and y<size[1]):
				result[x,y,:] = img_src[i,j,:]



	result[size[0]//2-img_dst.shape[0]//2:size[0]//2+img_dst.shape[0]//2, size[1]//2-img_dst.shape[1]//2:size[1]//2+img_dst.shape[1]//2,:] = img_dst[:,:,:]





	result_return = result.copy()

	for i in range(size[0]):
		for j in range(size[1]):
			if (result[i,j]==[0,0,0]).all():
				sumation1 = []
				sumation2 = []
				sumation3 = []
				for x in range(-3,4):
					for y in range(-3,4):
						if i+x>=0 and i+x < size[0] and j+y>=0 and j+y<size[1] and not(result[i+x,j+y,0]==0 and result[i+x,j+y,1]==0 and result[i+x,j+y,1]==0):
							sumation1.append(result[i+x,j+y,0])
							sumation2.append(result[i+x,j+y,1])
							sumation3.append(result[i+x,j+y,2])

				if len(sumation1)!=0 and len(sumation1)!=0 and len(sumation1)!=0:
					color1 = np.uint8(np.median(sumation1))
					color2 = np.uint8(np.median(sumation2))
					color3 = np.uint8(np.median(sumation3))
				else:
					color1 = 0
					color2 = 0
					color3 = 0

				result_return[i,j,0] = color1
				result_return[i,j,1] = color2
				result_return[i,j,2] = color3





	return result_return



print("loading data...")
image1 = cv2.imread('./test_sample4/test_sample4-1.jpg')
image2 = cv2.imread('./test_sample4/test_sample4-2.jpg')

# print('\nimage1\n')
# corners1 = findCorners(image1)
# print('\nimage2\n')
# corners2 = findCorners(image2)

# correspond12 = findCorrespondence(image1, corners1, image2, corners2, 70, 7000000)

# correspond12 = ransac(correspond12, 5, 10, np.pi/25, image1.shape[1])

correspond12 = [((80, 1429), (82, 910)), ((83, 1467), (86, 947)), ((94, 1574), (99, 1051)), ((100, 1678), (108, 1147)), ((149, 1689), (155, 1160)), ((159, 1509), (164, 993)), ((178, 1616), (360, 1449)), ((181, 1792), (188, 1253)), ((185, 1730), (191, 1199)), ((188, 1449), (194, 937)), ((198, 1269), (204, 753)), ((203, 1362), (360, 1449)), ((206, 1579), (211, 1064)), ((219, 1610), (86, 947)), ((221, 1679), (226, 1157)), ((247, 1350), (256, 842)), ((260, 1542), (266, 1033)), ((268, 1250), (279, 738)), ((272, 1678), (443, 1556)), ((274, 1789), (277, 1257)), ((278, 1351), (360, 1449)), ((280, 1211), (292, 698)), ((290, 1750), (352, 1231)), ((309, 1262), (322, 755)), ((312, 1207), (326, 696)), ((312, 1597), (587, 1666)), ((334, 1678), (344, 1163)), ((335, 1559), (340, 1054)), ((336, 1510), (343, 1008)), ((338, 1370), (349, 869)), ((342, 1791), (341, 1264)), ((352, 1752), (352, 1231)), ((359, 1900), (354, 1356)), ((360, 1432), (371, 933)), ((367, 1321), (381, 820)), ((384, 1778), (609, 1721)), ((386, 1219), (405, 715)), ((394, 1658), (395, 1149)), ((414, 1709), (414, 1197)), ((422, 1204), (444, 702)), ((426, 1517), (434, 1020)), ((427, 1743), (424, 1228)), ((434, 1815), (428, 1291)), ((444, 1650), (446, 1147)), ((467, 1870), (458, 1339)), ((476, 1399), (491, 908)), ((476, 1463), (488, 972)), ((477, 1105), (509, 597)), ((479, 1178), (506, 678)), ((491, 1671), (490, 1170)), ((497, 1249), (522, 756)), ((506, 1775), (500, 1262)), ((513, 1833), (502, 1312)), ((515, 1281), (572, 886)), ((538, 1907), (522, 1375)), ((545, 1109), (584, 606)), ((546, 1442), (560, 956)), ((548, 1762), (443, 1556)), ((553, 1370), (572, 886)), ((554, 1175), (509, 597)), ((561, 1407), (552, 1557)), ((572, 1880), (564, 1363)), ((575, 1729), (568, 1228)), ((581, 1786), (571, 1277)), ((589, 1600), (592, 1109)), ((594, 1449), (608, 966)), ((598, 1522), (607, 1038)), ((602, 962), (661, 440)), ((607, 1699), (602, 1202)), ((609, 1303), (689, 1440)), ((611, 1368), (632, 887)), ((613, 1418), (627, 935)), ((631, 1118), (509, 597)), ((634, 1518), (643, 1036)), ((640, 1823), (626, 1313)), ((679, 1007), (742, 500)), ((687, 1891), (665, 1373)), ((700, 1604), (703, 1123)), ((707, 1103), (864, 1222)), ((722, 1547), (731, 1071)), ((811, 1841), (789, 1343)), ((864, 1104), (931, 625)), ((875, 1698), (864, 1222)), ((898, 1107), (968, 632)), ((898, 1809), (877, 1323)), ((921, 1907), (887, 1406)), ((978, 1310), (1057, 1002)), ((1032, 1453), (1057, 1002)), ((1057, 1637), (1053, 1179))]
print(correspond12)

# print(correspond12)

# image1_color = image1
# image2_color = image2

# if image1_color.shape[0]>image2_color.shape[0]:
# 	tmp = np.zeros((image1_color.shape[0],image1_color.shape[1]+image2_color.shape[1]))
# 	tmp[:image1_color.shape[0],:image1_color.shape[1]] = image1_color[:,:]
# 	tmp[:image2_color.shape[0],image1_color.shape[1]:image1_color.shape[1]+image2_color.shape[1]] = image2_color[:,:]
# 	print("making result file...")
# 	print(image1_color.shape)
# 	print(tmp.shape)
# 	tmp = tmp.astype(np.uint8)
# 	result = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
# 	for i in correspond12:
# 		result = cv2.line(result, (i[0][1],i[0][0]), (i[1][1]+image1_color.shape[1],i[1][0]), (255,0,0), 2)

# else:
# 	tmp = np.zeros((image2_color.shape[0],image1_color.shape[1]+image2_color.shape[1]))
# 	tmp[:image1_color.shape[0],:image1_color.shape[1]] = image1_color[:,:]
# 	tmp[:image2_color.shape[0],image1_color.shape[1]:image1_color.shape[1]+image2_color.shape[1]] = image2_color[:,:]
# 	print("making result file...")
# 	print(image1_color.shape)
# 	print(tmp.shape)
# 	tmp = tmp.astype(np.uint8)
# 	result = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
# 	for i in correspond12:
# 		result = cv2.line(result, (i[0][1],i[0][0]), (i[1][1]+image1_color.shape[1],i[1][0]), (255,0,0), 2)

H = findHomography(correspond12, 50)

result = stiching(image2, image1, H, (5000,7000,3))


# tmp[:image1.shape[0],:image1.shape[1]] = image1[:,:]

cv2.imwrite("./result.png", result)