import numpy as np
import math
import cv2, PIL, os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def check_and_make_int(a):

	assert abs(a - int(a)) == False
	return int(a)


def add_guard_band_2D(greyimage, bandlengthx, bandlengthy):

	assert abs(bandlengthy - int(bandlengthy)) == False
	assert abs(bandlengthx - int(bandlengthx)) == False

	copy_image = []

	for row in greyimage:
		copy_image.append(list(row))

	for row_index in range(len(copy_image)):
		for i in range(int(bandlengthx)):
			copy_image[row_index].insert(0, 0)
			copy_image[row_index].append(0)

	row_length = len(copy_image[0])

	for i in range(int(bandlengthy)):
		copy_image.insert(0, row_length*[0])
		copy_image.append(row_length*[0])

	copy_image_array=np.array([np.array(row) for row in copy_image])

	return copy_image_array


def remove_guard_band_2D(image_with_guard, bandlengthx, bandlengthy):

	print('band length y, x',bandlengthy, bandlengthx)
	print('guarded image shape', image_with_guard.shape)
	removed_y_guard = image_with_guard[int(bandlengthy):-int(bandlengthy)]
	print('remved y guard shape', removed_y_guard.shape)
	new_image = []
	for row in range(len(removed_y_guard)):
		new_image.append(removed_y_guard[row][int(bandlengthx):-int(bandlengthx)])
		# new_image = removed_y_guard[:][int(bandlengthx):-int(bandlengthx)]
	new_image = np.asarray(new_image, dtype = np.float32)
	print(new_image.shape)
	return new_image


def transpose_matrix(matrix):

	print(type(matrix))

	assert type(matrix) != list

	shape = matrix.shape
	transpose_shape = (shape[1], shape[0])

	transposed_matrix = np.zeros(transpose_shape)

	for row_index in range(len(matrix)):
		row = matrix[row_index]
		for element_index in range(len(row)):
			transposed_matrix[element_index][row_index] = matrix[row_index][element_index]

	return transposed_matrix


def sampled_guassian(sigma):

	cutoff = 1/100
	
	samples = [1]

	for x in range(100):

		y = 1 / (np.sqrt(2*np.pi*sigma**2)) * math.exp( -0.5 * (x/sigma)**2)

		if y >= cutoff:
			samples.append(y)
			samples.insert(0, y)

		else:
			break

	normalised_samples = samples / sum(samples)
	print('-------------------------------')
	print('length',len(normalised_samples))
	# print(normalised_samples)
	# exit()
	return normalised_samples


def convolve_1D(function, filter_1D):

	reverse_filter = filter_1D[::-1]
	filter_length = len(reverse_filter)
	tail_length = 0.5*(filter_length-1)

	y = len(function) * [0]

	for i in range(int(tail_length), int(len(function)-tail_length-1), 1):

		segment = function[int(i-tail_length) : int(i+tail_length+1)]

		multiplied = segment*reverse_filter
		y[i] = int(sum(multiplied))

	return y


def convolve_2D(greyImage, filter_1D):

	filter_length = len(filter_1D)
	tail_length = 0.5*(filter_length-1)

	guarded_image = add_guard_band_2D(greyImage, tail_length, tail_length)
	x_convolved_image = guarded_image.copy()
	transposed_convolved_image = transpose_matrix(guarded_image.copy())

	print('dealing with rows')
	for i in range(len(guarded_image)):

		row = guarded_image[i]
		x_convolved_image[i] = convolve_1D(row, filter_1D)

	transposed_x_convolved_image = transpose_matrix(x_convolved_image)


	print('dealing with columns')
	for i in range(len(transposed_x_convolved_image)):

		column = transposed_x_convolved_image[i]
		transposed_convolved_image[i] = convolve_1D(column, filter_1D)

	convolved_image = transpose_matrix(transposed_convolved_image)
	convolved_image = remove_guard_band_2D(convolved_image, tail_length, tail_length)

	try:
		assert convolved_image.shape == greyImage.shape
	except AssertionError:
		print('convolved_image shape', convolved_image.shape)
		print('greyimage shape', greyImage.shape)

	return convolved_image


def downsample_2D(image, factor_1D):

	shape = image.shape
	new_x_len = shape[1]/factor_1D
	new_y_len = shape[0]/factor_1D
	
	try:
		assert abs(new_x_len - int(new_x_len)) == False
		new_x_len = int(new_x_len)
	except:
		new_x_len = math.floor((shape[1]-1)/factor_1D)
	try:
		assert abs(new_y_len - int(new_y_len)) == False
		new_y_len = int(new_y_len)
	except:
		new_y_len = math.floor((shape[0])/factor_1D)

	new_image = np.zeros([new_y_len, new_x_len])
	print(new_x_len, new_y_len)

	for i in range(new_y_len):
		for j in range(new_x_len):
			new_image[i][j] = image[factor_1D*i][factor_1D*j]

	return new_image


def downsample_2D_average(image, factor_1D):

	print('=> Downsampling')
	shape = image.shape
	old_x_len = shape[1]
	old_y_len = shape[0]

	new_x_len = shape[1]/factor_1D
	new_y_len = shape[0]/factor_1D

	factor_1D = check_and_make_int(factor_1D)

	try:
		assert abs(new_x_len - int(new_x_len)) == False
		new_x_len = int(new_x_len)
	except:
		new_x_len = math.ceil((shape[1])/factor_1D)
	try:
		assert abs(new_y_len - int(new_y_len)) == False
		new_y_len = int(new_y_len)
	except:
		new_y_len = math.ceil((shape[0])/factor_1D)

	x_downsampled_image = np.zeros([old_y_len, new_x_len])
	downsampled_image = np.zeros([new_y_len, new_x_len])
	print(new_x_len, new_y_len)

	for i in range(old_y_len):
		for j in range(new_x_len):
			average = 0
			try:
				for p in range(factor_1D):
					average += 1/factor_1D * image[i][j*factor_1D + p]
				x_downsampled_image[i][j] = average
			except IndexError:
				number_more_needed = new_x_len - old_x_len
				for p in range(number_more_needed):
					average += 1/number_more_needed * image[i][j*factor_1D + p]
				x_downsampled_image[i][j] = average

	for i in range(new_y_len):
		for j in range(new_x_len):
			average = 0
			try:
				for p in range(factor_1D):
					average += 1/factor_1D * x_downsampled_image[i*factor_1D + p][j]
				downsampled_image[i][j] = average
			except IndexError:
				number_more_needed = new_y_len - old_y_len
				for p in range(number_more_needed):
					average += 1/number_more_needed * image[i*factor_1D + p][j]
				x_downsampled_image[i][j] = average

	assert downsampled_image.shape[0] >= shape[0]/factor_1D
	assert downsampled_image.shape[1] >= shape[1]/factor_1D

	return downsampled_image


def upsample_2D(image, factor_1D, required_size):

	factor_1D = check_and_make_int(factor_1D)



	if factor_1D != int(1):
		print('in the if')
		old_shape = image.shape
		old_x_len = old_shape[1]
		old_y_len = old_shape[0]

		new_x_len = old_x_len*factor_1D
		
		new_y_len = old_y_len*factor_1D
		print(old_y_len, factor_1D)
		print(new_y_len)
		print(required_size)

		new_x_len = check_and_make_int(new_x_len)
		new_y_len = check_and_make_int(new_y_len)
		

		print('factor_1D', factor_1D, 'required_size', required_size)

		x_upsampled_image = np.zeros([old_y_len, new_x_len])
		upsampled_image = np.zeros([new_y_len, new_x_len])
		print(new_x_len, new_y_len)

		print()
		for i in range(old_y_len):
			for j in range(old_x_len-1):

				first_pixel = image[i][j]
				second_pixel_x = image[i][j+1]

				for p in range(factor_1D):
					
					fraction = 1/factor_1D
					x_upsampled_image[i][j*factor_1D+p] = first_pixel + fraction*(second_pixel_x-first_pixel)
			
			first_pixel = image[i][j+1]

			for p in range(factor_1D):
				
				fraction = 1/factor_1D
				x_upsampled_image[i][(j+1)*factor_1D+p] = first_pixel 



		for i in range(old_y_len-1):
			for j in range(new_x_len):

				first_pixel = x_upsampled_image[i][j]
				second_pixel_x = x_upsampled_image[i+1][j]

				for p in range(factor_1D):
					
					fraction = 1/factor_1D
					upsampled_image[i*factor_1D+p][j] = first_pixel + fraction*(second_pixel_x-first_pixel)
		
		first_row = x_upsampled_image[i+1][j]

		for p in range(factor_1D):
			
			fraction = 1/factor_1D
			upsampled_image[(i+1)*factor_1D+p] = first_row 


	else:
		print('in the else')
		upsampled_image = image

	size = upsampled_image.shape

	print(type(upsampled_image))
	try:
		assert size[0] == required_size[0]
	except:
		try:
			assert size[0] > required_size[0]
		except:
			print('Upsampled image smaller than required in y')
			print(size[0], '!>', required_size[0])
			exit()
		remove_band = size[0] - required_size[0]
		for i in range(remove_band):
			upsampled_image = upsampled_image[:-1]

	try:
		assert size[1] == required_size[1]
	except:
		try:
			assert size[1] > required_size[1]
		except:
			print('Upsampled image smaller than required in x')
			print(size[1], '!>', required_size[1])			
			exit()
		remove_band = size[1] - required_size[1]

		new_upsampled_image = np.zeros([required_size[0], required_size[1]])

		for i in range(len(upsampled_image)):
			new_upsampled_image[i] = upsampled_image[i][:-remove_band]
		upsampled_image = new_upsampled_image
		size = upsampled_image.shape
		assert size[0] == required_size[0]
		assert size[1] == required_size[1]

	return upsampled_image


def blur(image, sigma):
	print('sigma fed to blur', sigma)
	new_image = convolve_2D(image, sampled_guassian(sigma))
	return new_image


class IMAGE:

	def __init__(self, name, extension='.jpg'):

		self.name = name
		self.pathname = 'testImages/' + self.name + extension
		self.colour = cv2.imread(self.pathname)
		self.grey = cv2.cvtColor(self.colour, cv2.COLOR_BGR2GRAY)
		self.current = self.grey
		self.size = self.grey.shape
		self.total_blur_level = 0
		self.differential_level = 0
		self.downsample_level = 0


	def reset(self):
		self.current = self.grey


	def blur(self, sigma = 5):

		self.total_blur_level = np.sqrt(self.total_blur_level**2 + sigma**2)
		print('=> Blurring with sigma = %s for total blur of sigma = %s' % (sigma, self.total_blur_level))
		path_to_folder = 'cache/blurred/'
		filename = path_to_folder + '%s_sigma%s_down%s.jpg' % (self.name, \
															   self.total_blur_level, \
															   self.downsample_level)
		try:
			colour = cv2.imread(filename)
			self.current = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
		except:
			self.current = blur(self.current, sigma)
			cv2.imwrite(filename, self.current)
 

	def display(self, time=10):

		viewableimage = np.asarray(self.current, dtype=np.float32)/255
		cv2.imshow('Image', viewableimage)
		cv2.waitKey(time*1000)


	def differentiate(self):
		self.differential_level += 1
		path_to_folder = 'cache/differentiated/'
		filename = path_to_folder + '%s_level%s_down%s.jpg' % (self.name, self.differential_level, self.downsample_level)
		try:

			colour = cv2.imread(filename)
			self.current = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
			print('tried successfully')
		except:
			image = convolve_2D(self.grey, [-0.5, 0, 0.5])
			normalised_image = (image - np.amin(image)) * 255 / (np.amax(image) - np.amin(image))
			self.current = normalised_image
			cv2.imwrite(filename, self.current)
			print('excepted')


	def threshold(self, level = 180):
		image = self.current
		new_image = image.copy()
		for i in range(len(image)):
			for j in range(len(image[i])):

				if image[i][j] >= level:
					new_image[i][j] = 255
				else:
					new_image[i][j] = 0

		self.current = new_image


	def downsample_2D(self, factor_1D = 2):
		
		new = downsample_2D(self.current, factor_1D)
		print(new.shape, self.current.shape)
		self.current = new
		print('hsdgffae.rkhgea.krgh',self.current.shape)


	def __collect_images_for_bank(self, octaves, levels, file_name):

		assert(os.path.exists(file_name))
		bank = []
		for octave in range(octaves):
			one_octave = []
			for level in range(levels-1):
				image = cv2.imread(file_name + '/octave_%s_level_%s.jpg' % (octave, level))
				grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				# print('octave level', octave, level)
				one_octave.append(grey_image)
				# print(len(image_bank[octave][level]))
				# print(len(image_bank), len(image_bank[0]),len(image_bank[0][2]), len(image_bank[0][0][0]))
			bank.append(one_octave)
		return bank


	def __create_laplacian_pyramid_1st(self, octaves, levels, sigma0):

		laplacian_bank = []
		blur_bank = self.image_pyramid
		laplacian_pyramid_filename = 'cache/%s/laplacian_pyramid_%s_%s' % (self.name, octaves, levels)

		try:
			assert(os.path.exists(laplacian_pyramid_filename))

			for octave in range(octaves):
				one_octave = []
				for level in range(levels-1):
					image = cv2.imread(laplacian_pyramid_filename + '/octave_%s_level_%s.jpg' % (octave, level))
					grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
					# print('octave level', octave, level)
					one_octave.append(grey_image)
					# print(len(image_bank[octave][level]))
					# print(len(image_bank), len(image_bank[0]),len(image_bank[0][2]), len(image_bank[0][0][0]))
				laplacian_bank.append(one_octave)
		except AssertionError:
			os.mkdir(laplacian_pyramid_filename)

			laplacian_bank = []

			for octave in range(octaves):
				current_octave = []
				for level in range(levels-1):
					print('-------------')
					# print(image_bank[octave][level], image_bank[octave][level+1])


					image = np.asarray(blur_bank[octave][level], dtype = np.float32)
					next_image = np.asarray(blur_bank[octave][level+1], dtype = np.float32)
					
					rows = image.shape[0]
					columns = image.shape[1]

					laplacian_image = np.zeros([rows, columns])

					laplacian_image = next_image - image

					print(type(next_image[0][0]))
					print('pixel image1: ', next_image[0][0], ' - pixel image2: ', image[0][0], '= pixel laplacian: ', next_image[0][0] - image[0][0])

					# for row in range(rows):
					# 	for pixel in range(columns):

					# 		laplacian_image[row][pixel] = next_image[row][pixel] - image[row][pixel]
					print(laplacian_image)
					normalised_image = (laplacian_image - np.amin(laplacian_image)) * 255 / (np.amax(laplacian_image)-np.amin(laplacian_image))
					current_octave.append(normalised_image)

					cv2.imwrite(laplacian_pyramid_filename + '/octave_%s_level_%s.jpg' % (octave, level), normalised_image)
				
				# if octave != octaves-1:
				# 	print('herehrerererere')
				# 	image = np.asarray(blur_bank[octave][-1], dtype = np.float32)
				# 	next_image = np.asarray(blur_bank[octave+1][0], dtype = np.float32)
				# 	laplacian_image = np.zeros([rows, columns])
				# 	laplacian_image = next_image - image	
				# 	normalised_image = (laplacian_image - np.amin(laplacian_image)) * 255 / (np.amax(laplacian_image)-np.amin(laplacian_image))
				# 	current_octave.append(normalised_image)
				# 	cv2.imwrite(laplacian_pyramid_filename + '/octave_%s_level_%s.jpg' % (octave, level+1), normalised_image)


				laplacian_bank.append(current_octave)

		return laplacian_bank


	def __create_laplacian_pyramid_2nd(self, octaves, levels, sigma0):

		laplacian_bank = []
		blur_bank = self.upsampled_blurred
		laplacian_pyramid_filename = 'cache/%s/laplacian_pyramid_%s_%s' % (self.name, octaves, levels)

		try:
			assert(os.path.exists(laplacian_pyramid_filename))

			for octave in range(octaves):
				one_octave = []
				for level in range(levels-1):
					image = cv2.imread(laplacian_pyramid_filename + '/octave_%s_level_%s.jpg' % (octave, level))
					grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
					# print('octave level', octave, level)
					one_octave.append(grey_image)
					# print(len(image_bank[octave][level]))
					# print(len(image_bank), len(image_bank[0]),len(image_bank[0][2]), len(image_bank[0][0][0]))
				laplacian_bank.append(one_octave)
		except AssertionError:
			os.mkdir(laplacian_pyramid_filename)

			laplacian_bank = []

			for octave in range(octaves):
				current_octave = []
				for level in range(levels-1):
					print('-------------')
					# print(image_bank[octave][level], image_bank[octave][level+1])


					image = np.asarray(blur_bank[octave][level], dtype = np.float32)
					next_image = np.asarray(blur_bank[octave][level+1], dtype = np.float32)
					
					rows = image.shape[0]
					columns = image.shape[1]

					laplacian_image = np.zeros([rows, columns])

					laplacian_image = next_image - image

					print(type(next_image[0][0]))
					print('pixel image1: ', next_image[0][0], ' - pixel image2: ', image[0][0], '= pixel laplacian: ', next_image[0][0] - image[0][0])

					# for row in range(rows):
					# 	for pixel in range(columns):

					# 		laplacian_image[row][pixel] = next_image[row][pixel] - image[row][pixel]
					print(laplacian_image)
					normalised_image = (laplacian_image - np.amin(laplacian_image)) * 255 / (np.amax(laplacian_image)-np.amin(laplacian_image))
					current_octave.append(normalised_image)

					cv2.imwrite(laplacian_pyramid_filename + '/octave_%s_level_%s.jpg' % (octave, level), normalised_image)
				
				# if octave != octaves-1:
				# 	print('herehrerererere')
				# 	image = np.asarray(blur_bank[octave][-1], dtype = np.float32)
				# 	next_image = np.asarray(blur_bank[octave+1][0], dtype = np.float32)
				# 	laplacian_image = np.zeros([rows, columns])
				# 	laplacian_image = next_image - image	
				# 	normalised_image = (laplacian_image - np.amin(laplacian_image)) * 255 / (np.amax(laplacian_image)-np.amin(laplacian_image))
				# 	current_octave.append(normalised_image)
				# 	cv2.imwrite(laplacian_pyramid_filename + '/octave_%s_level_%s.jpg' % (octave, level+1), normalised_image)


				laplacian_bank.append(current_octave)

		return laplacian_bank


	def __create_image_pyramid(self, octaves, levels, sigma0):
		
		image_bank = []
		
		image_pyramid_filename = 'cache/%s/image_pyramid_%s_%s' % (self.name, octaves, levels)


		try:
			assert(os.path.exists(image_pyramid_filename))

			for octave in range(octaves):

				one_octave = []

				for level in range(levels):
					image = cv2.imread(image_pyramid_filename + '/octave_%s_level_%s.jpg' % (octave, level))
					grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
					one_octave.append(grey_image)
				image_bank.append(one_octave)

		except AssertionError:
			os.mkdir(image_pyramid_filename)
			
			print('excepting')
			sigma_bank = np.zeros(levels)
			sigma_bank[0] = sigma0
			for i in range(len(sigma_bank)-1):
				incremental_sigma = sigma_bank[i] * np.sqrt(2**(2/levels)-1)
				sigma_bank[i+1] = incremental_sigma
			
			
			original_image = self.grey.copy()
			working_image = original_image.copy()
			

			for octave in range(octaves):
				filename = image_pyramid_filename + '/octave_%s_level_0.jpg' % octave
				cv2.imwrite(filename, working_image)

				one_octave=[]
				one_octave.append(working_image)

				for level in range(levels-1):
					filename = image_pyramid_filename + '/octave_%s_level_%s.jpg' % (octave, level+1)
					sigma = sigma_bank[level]
					next_image = blur(working_image, sigma)
					one_octave.append(next_image)
					working_image = next_image
					cv2.imwrite(filename, working_image)
				image_bank.append(one_octave)
				working_image = downsample_2D_average(working_image, 2)

		return image_bank


	def __create_upsampled_blur_pyramid(self, octaves, levels):

		upsampled_bank = []

		image_pyramid_filename = 'cache/%s/upsampled_blur_pyramid_%s_%s' % (self.name, octaves, levels)

		try:
			assert(os.path.exists(image_pyramid_filename))

			for octave in range(octaves):

				one_octave = []

				for level in range(levels):
					image = cv2.imread(image_pyramid_filename + '/octave_%s_level_%s.jpg' % (octave, level))
					grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
					one_octave.append(grey_image)
				upsampled_bank.append(one_octave)

		except:
			os.mkdir(image_pyramid_filename)

			blur_bank = self.image_pyramid
			blur_pyramid_shape = (len(blur_bank), len(blur_bank[0]))

			required_size = self.size

			for octave in range(blur_pyramid_shape[0]):
				one_octave = []
				for level in range(blur_pyramid_shape[1]):

					downsampled_level = (2**octaves)/(2**(octaves-octave))
					print('=> Upsampling octave %s level %s by factor of %s' % (octave, level, downsampled_level))
					print(len(blur_bank[-1][-1]), len(blur_bank[-1][-1][0]))
					upsampled_image = upsample_2D(blur_bank[octave][level], downsampled_level, required_size)
					filename = image_pyramid_filename + '/octave_%s_level_%s.jpg' % (octave, level)
					cv2.imwrite(filename, upsampled_image)
					one_octave.append(upsampled_image)
				upsampled_bank.append(one_octave)

		return upsampled_bank


	def detect_blobs(self, octaves, levels):

		image_pyramid_filename = 'cache/%s/blobs_%s_%s' % (self.name, octaves, levels)

		try:
			os.mkdir(image_pyramid_filename)
		except:
			pass

		blob_bank = []
		laplacian_levels = levels-1

		laplacian_bank = self.laplacian_pyramid
		flat_laplacian_bank = []

		for octave in range(octaves):
			for level in range(laplacian_levels):

				flat_laplacian_bank.append(laplacian_bank[octave][level])

		laplacian_set_size = len(flat_laplacian_bank)

		# for scale in range(1, laplacian_set_size-1):
		for scale in range(36, 38):
			print('scale: ', scale)

			image = flat_laplacian_bank[scale]
			view_image = image.copy()
			self.colour.copy()

			for i in range(1, len(image)-1):

				for j in range(1, len(image[0])-1):

					pixel = image[i][j]
					left_pixel = image[i][j-1]
					right_pixel = image[i][j+1]
					print(pixel)

					if pixel >= left_pixel and pixel >= right_pixel:

						above_left_pixel = image[i-1][j-1]
						above_pixel = image[i-1][j]
						above_right_pixel = image[i-1][j+1]
						bellow_left_pixel = image[i+1][j-1]
						bellow_pixel = image[i+1][j]
						bellow_right_pixel = image[i+1][j+1]

						pixels = [above_pixel, above_right_pixel, above_left_pixel, \
								  bellow_pixel, bellow_right_pixel, bellow_left_pixel, right_pixel, left_pixel]

						if scale == 37:
							print(i, j, pixel, pixels)

						if pixel > np.amax(pixels):



							prev_image = flat_laplacian_bank[scale-1]
							next_image = flat_laplacian_bank[scale+1]

							prev_pixels = []
							next_pixels = []

							for p in range(-1, 2):
								for q in range(-1, 2):
									if p !=0 or q != 0:
										prev_pixels.append(prev_image[i+p][j+q])
										next_pixels.append(next_image[i+p][j+q])

							if pixel > np.amax(prev_pixels) and pixel >= np.amax(next_pixels):


								blob_bank.append([i, j, scale])

								cv2.circle(view_image,(i, j), scale, (0,255,0), 1)

			cv2.imwrite(image_pyramid_filename + "/scale_%s.jpg" % scale, view_image)


			# print (blob_bank)


	def detect_blobs_2(self, octaves, levels):

		image_pyramid_filename = 'cache/%s/blobs_%s_%s' % (self.name, octaves, levels)

		try:
			os.mkdir(image_pyramid_filename)
		except:
			pass

		blob_bank = []
		laplacian_levels = levels-1

		laplacian_bank = self.laplacian_pyramid
		flat_laplacian_bank = []
		for octave in range(octaves):
			# image = laplacian_bank[octave][0]
			# for row in range(1, len(image)):
			# 	for column in range(1, len(image[0])):
			# 		pixel = image[row][column]
			# 		pixels = []
			# 		for p in range(-1, 2):
			# 			for q in range(-1, 2):
			# 				if p!=0 or q!=0:
			# 					pixels.append(image[row+q][pixel+q])
			# 		if pixel > np.amax(pixels):
			# 			next_image = laplacian_bank[0][1]
			# 			next_pixels = []
			# 			for p in range(-1, 2):
			# 				for q in range(-1, 2):
			# 					next_pixels.append(next_image[row+q][pixel+q])
			# 			if pixel > np.amax(next_pixels):
			# 				blob_bank.append([row, column, octave, level])
			# 				cv2.circle(image,(row, column), 5, (0,255,0), 1)
			# cv2.imwrite(image_pyramid_filename + "/octace_%s_level_%s.jpg" % (octave, 0))

			for level in range(laplacian_levels):
				image = laplacian_bank[octave][level]
				colour_image = np.zeros([len(image), len(image[0]), 3])

				print(len(image))
				print(len(image[0]))

				for row in range(len(image)):
					for column in range(len(image[0])):
						pixel = image[row][column]
						colour_image[row][column] = [pixel, pixel, pixel]

				for row in range(1, len(image)-1):
					for column in range(1, len(image[row])-1):
						# print(row, column)
						pixel = image[row][column]
						surrounding_pixels = []
						for p in range(-1, 2):
							for q in range(-1, 2):
								if p!=0 or q!=0:
									surrounding_pixels.append(image[row+p][column+q])
						# if pixel >= np.amax(surrounding_pixels):
						# 	# print(pixel, surrounding_pixels)
						if pixel >= np.amax(surrounding_pixels):
							# print('got so far')
							try:
								next_image = laplacian_bank[octave][level+1]
							except IndexError:
								pass
							next_pixels = []
							try:
								prev_image = laplacian_bank[octave][level-1]
							except IndexError:
								pass
							prev_pixels = []
							for p in range(-1, 2):
								for q in range(-1, 2):
									try:
										surrounding_pixels.append(next_image[row+p][column+q])
									except:
										pass
									try:
										surrounding_pixels.append(prev_image[row+p][column+q])
									except:
										pass
							assert len(surrounding_pixels) > 8							# print(np.amax(surrounding_pixels))
							if pixel>=np.amax(surrounding_pixels) or pixel<=np.amin(surrounding_pixels):

								blob_bank.append([row, column, octave, level])
								cv2.circle(colour_image,(column, row), int(len(image)/75), (0,255,0), 1)
				cv2.imwrite(image_pyramid_filename + "/octace_%s_level_%s.jpg" % (octave, level), colour_image)
		print(blob_bank)

		f= open(image_pyramid_filename + "/blob_bank.txt","w+")
		f.write(str(blob_bank))
		return


	def blob_detector(self, octaves=6, levels=8, start_sigma = 1):
		
		try:
			os.mkdir('cache/%s' % self.name)
		except:
			pass

		try:
			assert levels > 1
			assert octaves > 1 
		except:
			print('Must have both octave and levels >1')
			exit()
		print(octaves, levels, start_sigma)
		self.image_pyramid = self.__create_image_pyramid(octaves=octaves, levels=levels, sigma0=start_sigma)
		self.laplacian_pyramid = self.__create_laplacian_pyramid_1st(octaves=octaves, levels=levels, sigma0=start_sigma)
		self.detect_blobs_2(octaves, levels)

		# self.upsampled_blurred = self.__create_upsampled_blur_pyramid(octaves=octaves, levels=levels)	
		# self.laplacian_upsampled = self.__create_upsampled_lapl_pyramid(octaves)
		# self.laplacian_pyramid = self.__create_laplacian_pyramid_2nd(octaves=octaves, levels=levels, sigma0=start_sigma)
		# self.upsampled_lapl_pyramid = self.__create_upsampled_lapl_pyramid(octaves=octaves, levels=levels)
		# self.current = upsample_2D(laplacian_bank[-1][-1], 2)
		# self.detect_blobs(octaves=octaves, levels=levels)



image5_1 = IMAGE('image5_2')
image13 = IMAGE('image13', extension='.png')

image12 = IMAGE('image12')
# image5_1.blob_detector()
# image12.blob_detector(start_sigma=1)
image12.blob_detector(start_sigma=1)


