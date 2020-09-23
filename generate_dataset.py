import cv2
import numpy as np
import json
import math
import os
import argparse
import time
import shutil
from tqdm import tqdm
import multiprocessing as mp
from joblib import Parallel, delayed
from imgaug import augmenters as iaa
import imgaug as ia
from matplotlib import pyplot as plt
import pickle
# import msgpack as pickle
# import msgpack_numpy as m
# m.patch()

# Seed sendo setada como None em generate_sample
# np.random.seed(2018)

def augment_target(target, multiply_value=None, add_value=None):
	if add_value is None:
		add_value = int(np.random.uniform(-120, 120))
	if multiply_value is None:
		multiply_value = float(np.random.uniform(0.75, 1.25))
	seq = iaa.Sequential([
		iaa.Add((add_value, add_value)),
		iaa.Multiply((multiply_value, multiply_value)),
		# iaa.GaussianBlur(blur)
	])
	target = (target * 255.0).astype(np.uint8)
	return (seq.augment_image(target) / 255.0).astype(np.float32), add_value, multiply_value


def test_get_position():
	with open("annotations-pretty.json", "r") as annotations_f:
		annotations = json.load(annotations_f)
	probabilities_vector = get_probabilties_vector(annotations)
	positions_list = np.arange(0, probabilities_vector.size)
	xy = get_position(probabilities_vector,
							 positions_list, sample_size=100000)
	mapa = np.zeros((2048, 2048))
	map_size = (2048, 2048)
	for idx in range(len(xy)):
		x, y = xy[idx]
		x -= 5
		y -= 5
		xmax = x + 10
		ymax = y + 10

		if x < 0:
			x = 0
		if y < 0:
			y = 0

		if xmax > map_size[0]:
			xmax = map_size[0]
		if ymax > map_size[1]:
			ymax = map_size[1]
		mapa[y:ymax, x:xmax] = np.add(
			mapa[y:ymax, x:xmax], np.full((ymax - y, xmax - x), 1))
	plt.imshow(mapa, cmap='viridis', interpolation='nearest')
	plt.show()


def blur(img, mask, target_region, value=None):
	if value is None:
		value = float(np.random.uniform(0, 2.75))
	blur_effect = iaa.Sequential([iaa.GaussianBlur(value)])
	cpy = target_region.copy()
	cpy[mask[:,:,0] > 0] = img[:,:,[0,1,2]][mask[:,:,0] > 0]
	img = blur_effect.augment_image(cpy)
	return img
	# return cv2.GaussianBlur(img, (5, 5), 0), cv2.GaussianBlur(mask, (5, 5), 0)


def histogram_noise(img, template_mask, noise=(-15, 15), data=None):
	if data is not None:
		np.random.set_state(data['state'])
	noise = np.random.randint(noise[0], noise[1], size=img.shape)
	return np.clip(img + (noise / 255.0), 0, 1), {'state':  np.random.get_state()}


def brightness_transform(template, add_value, multiply_value):
	template, add_value, multiply_value = augment_target(template, multiply_value, add_value + 40)
	return template, {}

def geometric_transform(template, template_mask, x, target_size, scale=None, prelodaded_data=None):
	data = {}
	rows, cols, _ = template.shape
	x -= int(round(target_size[0] / 2))
	relative_x = abs(2 * x / target_size[0])

	# # perspective
	persp_max_min = (
		int(round(template.shape[1] * 0.07)), int(round(template.shape[1] * 0.14)))
	h,w, _ = template.shape
	persp_min, persp_max = persp_max_min
	if prelodaded_data is None:
		persp = np.random.randint(persp_min, persp_max + 1)
	else:
		persp = prelodaded_data['persp']
	data['persp'] = persp

	if x > 0:
		pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
		pts2 = np.float32([[persp, persp], [w - 1, 0],
						[persp, h - 1 - persp], [w - 1, h - 1]])
	else:
		pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
		pts2 = np.float32([[0, 0], [w - 1 - persp, persp],
						[0, h - 1],[w - 1 - persp, h - 1 - persp]])

	M = cv2.getPerspectiveTransform(pts1, pts2)
	template = cv2.warpPerspective(template, M, (cols, rows))
	template_mask = cv2.warpPerspective(
		template_mask, M, (cols, rows))

	# rotation
	if prelodaded_data is None:
		angle = np.random.uniform(-2, 2)
	else:
		angle = prelodaded_data['angle']
	data['angle'] = angle
	M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
	template = cv2.warpAffine(template, M, (cols, rows), flags=cv2.INTER_LINEAR)
	template_mask = cv2.warpAffine(template_mask, M, (cols, rows), flags=cv2.INTER_LINEAR)

	# scale
	if prelodaded_data is None:
		if scale is None:
			scale_factor = (.175 + relative_x) * (min(template.shape[0], template.shape[1])) / 100.0
		else:
			scale_factor = scale
	else:
		scale_factor = prelodaded_data['scale']
	data['scale'] = scale_factor
	'''template = cv2.resize(template, (0, 0), fx=scale_factor, fy=scale_factor)
	template_mask = cv2.resize(
		template_mask, (0, 0), fx=scale_factor, fy=scale_factor)'''

	# template_mask[template_mask < 0.5] = 0

	return template, template_mask, data

def blend(template, template_mask, target_image, steps=3):
	template = (template * 255).astype(np.uint8)
	target_image = (target_image * 255).astype(np.uint8)

	temp_template_mask = template_mask.copy()
	temp_template_mask = cv2.copyMakeBorder(temp_template_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0)) # add padding pra evitar bug de no-erode nas bordas
	
	blend_mask = temp_template_mask.astype(np.float32) * (1.0 / steps)
	kernel = np.ones((3, 3), np.uint8)

	for step in range(steps-1):
		temp_template_mask = cv2.erode(temp_template_mask, kernel)
		blend_mask += temp_template_mask * (1.0 / steps)

	blend_mask = blend_mask[1:-1, 1:-1] # remove padding
	blended = (target_image * (1 - blend_mask)) + (template[:,:,[0,1,2]] * blend_mask)
	
	return blended.astype(np.float32) / 255.0
	


# def blend_rodrigo(template, template_mask, target_image, target_bbox):
#	 cv2.imshow('template do blend', template)
#	 cv2.imshow('template mask do blend', template_mask)
#	 bbox_h, bbox_w = template_mask.shape[:-1]
#	 bbox_img = target_image[target_bbox['y0']:target_bbox['y1'],
#							 target_bbox['x0']:target_bbox['x1'], :]
#	 #print(bbox_img.shape, bbox_h, bbox_h, target_bbox)
#	 assert bbox_img.shape[0] == bbox_h
#	 assert bbox_img.shape[1] == bbox_w
#	 assert len(np.unique(template_mask)) == 2

#	 # create the blend mask
#	 # blended_naive = bbox_img.copy()
#	 # blended_naive[template_mask > 0] = template[..., :3][template_mask > 0]
#	 template_mask = (template_mask[..., 0] > 0).astype(np.float32)
#	 kernel = np.ones((3, 3), np.uint8)
#	 blending_mask = np.zeros_like(template_mask)
#	 # if min(bbox_h, bbox_w) < 30:
#	 #	 p_step = 0.6
#	 # else:
#	 #	 p_step = 0.275
#	 p_step = 0.275
#	 for i, p in enumerate(np.arange(0, 1, p_step).tolist() + [1]):
#		 erosion = cv2.erode(template_mask, kernel, iterations=i) * p
#		 blending_mask[erosion > 0] = erosion[erosion > 0]
#	 blending_mask_3c = np.repeat(blending_mask[..., np.newaxis], 3, axis=-1)
#	 blended = bbox_img.copy()
#	 blended *= (1 - blending_mask_3c)
#	 blended += (template[..., :3] * blending_mask_3c)
#	 cv2.imshow('blending mask', blending_mask_3c)
#	 blended = np.clip(blended, 0., 1.)

#	 return blended


def get_position(position, img_size=(2048, 2048)):
	x = position % img_size[1]
	y = int((position - x)/img_size[1])

	return (x, y)


def get_probabilties_vector(annotations_path, map_size=(2048, 2048)):
	def get_annotations_positions_vector_germany(annotations_path):
		with open(annotations_path, "r") as annotations_f:
			annotations = json.load(annotations_f)

		lista = []
		for img_id in annotations['imgs']:
			for obj in annotations['imgs'][img_id]['objects']:
				xmin = int(obj['bbox']['xmin'])
				ymin = int(obj['bbox']['ymin'])
				xmax = int(obj['bbox']['xmax'])
				ymax = int(obj['bbox']['ymax'])

				if xmin < 0:
					xmin = 0
				if ymin < 0:
					ymin = 0

				if xmax > map_size[0]:
					xmax = map_size[0]
				if ymax > map_size[1]:
					ymax = map_size[1]

				lista.append((xmin, ymin, xmax, ymax))
		return lista
	def get_annotations_positions_vector_belgium(annotations_path):
		lista = []
		with open(annotations_path, "r") as annotations_f:
			lines = annotations_f.readlines()
		for line in lines:
			# 01/image.000935.jp2;1346.82;246.76;1582.12;484.41;65;2;
			path, xmin, ymin, xmax, ymax, class_id, superclass_id = line.rstrip().split(";")[:-1]
			xmin, ymin, xmax, ymax = (min(max(int(float(xmin)),0),map_size[0]), min(max(int(float(ymin)),0),map_size[1]), min(max(int(float(xmax)),0),map_size[0]), min(max(int(float(ymax)),0),map_size[1]))
			lista.append((xmin, ymin, xmax, ymax))
		return lista
	mapa = np.zeros(map_size[::-1])
	positions_vector = get_annotations_positions_vector_germany(annotations_path)
	for idx in range(len(positions_vector)):
		x, y, xmax, ymax = positions_vector[idx]
		#print(x,y,xmax,ymax)
		mapa[y:ymax, x:xmax] = np.add(
			mapa[y:ymax, x:xmax], np.full((ymax - y, xmax - x), 1))
	# mapa *= mapa
	# plt.imshow(mapa, cmap='viridis', interpolation='nearest')
	# if plt.waitforbuttonpress():
	#	 exit()
	probabilities_vector = mapa.flatten() / mapa.sum()

	return probabilities_vector


def multiply(img, multiply_value):
	aug = iaa.Multiply((multiply_value, multiply_value))
	img = (img * 255.0).astype(np.uint8)
	return (aug.augment_image(img) / 255.0).astype(np.float32)


def has_intersection(x0, y0, x1, y1, bboxes):
	for bbox in bboxes:
		if x1 > bbox['xmin'] and bbox['xmax'] > x0:
			if y1 > bbox['ymin'] and bbox['ymax'] > y0:
				return True
	return False

def remove_padding(template, template_mask):
	mask = template_mask[:,:,0] > 0

	# Coordinates of non-black pixels.
	coords = np.argwhere(mask)

	# Bounding box of non-black pixels.
	y0, x0 = coords.min(axis=0)
	y1, x1 = coords.max(axis=0) + 1   # slices are exclusive at the top

	# Get the contents of the bounding box.
	# cropped = image[y0:y1, x0:x1]
	return template[y0:y1, x0:x1], template_mask[y0:y1, x0:x1], x0, y0, x1, y1


def distribution(target):
	height_range_prob = 100 * np.random.random()
	if height_range_prob <= 0.9341075485988387:
		h = [0.09583333333333334, 0.16041666666666668]
		pos = [0, 0.1]
	elif height_range_prob <= 0.9341075485988387 + 10.12370613481444:
		h = [0.06666666666666667, 0.1625]
		pos = [0.1, 0.2]
	elif height_range_prob <= 0.9341075485988387 + 10.12370613481444 + 29.512749305730875:
		h = [0.03854166666666667, 0.10833333333333334]
		pos = [0.2, 0.3]
	elif height_range_prob <= 0.9341075485988387 + 10.12370613481444 + 29.512749305730875 + 55.76874526634688:
		h = [0.017708333333333333, 0.09791666666666667]
		pos = [0.3, 0.4]
	else:
		h = [0.017708333333333333, 0.03229166666666667]
		pos = [0.4, 0.5]

	return int(target.shape[0] * np.random.choice(np.random.uniform(h[0], h[1], 1000))), int(target.shape[0] * np.random.choice(np.random.uniform(pos[0], pos[1], 1000))) 

def process_img(target, traffic_scene_path, template_is_tf, tmpt_blur_factor, probabilities_vector, info_list, add_value, multiply_value, use_normal_dist, temp_min_h, temp_max_h, mean_heights, std_dev_heights, position=None, bboxes=None, show_template=False, blur_value=None, scale=None, prelodaded_data=None):	
	data = {}
	
	traffic_scene = cv2.imread(traffic_scene_path, cv2.IMREAD_UNCHANGED) / 255.0
	traffic_scene_mask = get_mask_from_image(traffic_scene)
	traffic_scene = cv2.resize(traffic_scene, (target.shape[1], target.shape[0]))
	traffic_scene_mask = cv2.resize(traffic_scene_mask, (target.shape[1], target.shape[0]))
	position_is_centered = True
	
	traffic_scene[:,:,0][traffic_scene_mask[:,:,0] < 1] = 0
	traffic_scene[:,:,1][traffic_scene_mask[:,:,0] < 1] = 0
	traffic_scene[:,:,2][traffic_scene_mask[:,:,0] < 1] = 0
	
	traffic_scene, brightness_transform_data = brightness_transform(traffic_scene, add_value, multiply_value)
	data['brightness_transform_data'] = brightness_transform_data

	traffic_scene, histogram_noise_data = histogram_noise(traffic_scene, traffic_scene_mask, data=prelodaded_data['histogram_noise_data'] if prelodaded_data is not None else None)
	data['histogram_noise_data'] = histogram_noise_data

	blur_effect = iaa.Sequential([iaa.GaussianBlur(tmpt_blur_factor, deterministic=True)])
	traffic_scene = blur_effect.augment_image(traffic_scene)
	
	target = blend(traffic_scene, traffic_scene_mask, target)
	
	bboxes = []
	for i, info in enumerate(info_list):
		if prelodaded_data is None:
			position = get_position(info['pos'], target.shape[:-1])
		else:
			position_is_centered = prelodaded_data['position_is_centered']
			position = prelodaded_data['positions'][i]
		
		if 'positions' not in data.keys():
			data['positions'] = []
		data['positions'].append((int(position[0]), int(position[1])))
		data['position_is_centered'] = position_is_centered
		x, y = position
		w, h = info['(w,h)']
		target_h, target_w, _ = target.shape
		
		if not position_is_centered:
			x0 = x
		else:
			x0 = int(round(x - w / 2))
		if x0 < 0:
			w += x0			
			x0 = 0
		if not position_is_centered:
			y0 = y
		else:
			y0 = int(round(y - h / 2))
		if y0 < 0:
			h += y0
			y0 = 0
		x1 = x0 + w
		y1 = y0 + h
		if x1 >= target_w:
			x1 = target_w - 1
		if y1 >= target_h:
			y1 = target_h - 1

		bboxes.append({
			'xmin': x0,
			'ymin': y0,
			'xmax': x1,
			'ymax': y1,
			'class': info['class'],
			'category': os.path.basename(traffic_scene_path).replace('.png', '')
		})

	return target, bboxes, data


def get_mask_from_image(alpha_image):
	alpha_channel = alpha_image[:, :, -1]
	mask = np.zeros_like(alpha_image[:, :, :-1])
	mask[:, :, 0][alpha_channel > 0] = 1
	mask[:, :, 1][alpha_channel > 0] = 1
	mask[:, :, 2][alpha_channel > 0] = 1

	return mask


def parse_args():
	parser = argparse.ArgumentParser(
		description='Generate a templated dataset.')

	parser.add_argument('--bgs-path', dest='targets_path', type=str, required=True,
						help='Path to the directory containing background images to be used.')
	parser.add_argument('--traffic-path', dest='traffic_path', type=str, required=True,
						help='Path to the directory containing artificial traffic contexts.')
	parser.add_argument('--labels-path', dest='labels_path', type=str, required=True,
						help='Path to the directory containing the traffic-lights labels.')
	parser.add_argument('--out-path', dest='out_path', type=str, required=True,
						help='Path to the directory to save the images generated to.')
	parser.add_argument('--total-images', dest='total_images', type=int, required=True,
						help='Number of images to be generated.')
	#parser.add_argument('--annotations-path', dest='annotations_path', type=str, required=True,
	#					help='Path to the annotations file.')
	parser.add_argument('--data', dest='random_data', type=str, default=None)
	parser.add_argument('--resize', dest='do_resize', action='store_true')
	parser.add_argument('--normal', dest='use_normal_dist', action='store_true')
	args = parser.parse_args()

	return args

def hc_probabilties_vector(map_size):
	mapa = np.ones(map_size)
	return mapa.flatten()/mapa.sum()

def show_grid_of_augs(templates):
	multiply_values = np.arange(0.75, 1.25, step=(1.25-0.75)/10)
	blur_values = np.arange(0, 7, step=(7-0)/10)
	targets = [("coco/selection/000000000294.jpg",(220,1220)),("coco/selection/000000000382.jpg",(770,900)),("coco/selection/000000000761.jpg",(130,1260)),("coco/selection/000000001111.jpg",(640,515))]

	assert(len(multiply_values) == 10)
	assert(len(blur_values) == 10)

	for template in templates:
		template = template[0]
		view = np.zeros((template.shape[0]*10,template.shape[1]*10, 3))
		for l, multiply_value in enumerate(multiply_values):
			for c, blur_value in enumerate(blur_values):
				aug_template = template.copy()

				aug_template = multiply(aug_template, multiply_value)

				target_region_min = np.ones((template.shape[0], template.shape[1], 3)) * 0

				aug_template = brightness_transform(aug_template, np.ones_like(template), target_region_min, None, use_fixed=60/255.0)


				aug_template = blur(aug_template, np.ones_like(template), target_region_min, value=blur_value)


				# aug_template = histogram_noise(aug_template, np.ones_like(template))
				x = template.shape[1]*c
				x2 = x + template.shape[1]

				y = template.shape[0]*l
				y2 = y + template.shape[0]
				view[y:y2, x:x2, :] = aug_template
				print("({},{}) = ({},{})   = {} {}".format(l,c,multiply_value,blur_value, x, y))


		for img_path, pos in targets:
			y, x = pos
			target = cv2.imread(img_path).astype(np.float32) / 255.0
			aug_template = template.copy()
			result, _, _ = process_img(target.copy(), aug_template, get_mask_from_image(aug_template),hc_probabilties_vector((target.shape[1], target.shape[0])),None, multiply_value=multiply_values[0], position=pos[::-1], blur_value=blur_values[-1])
			cv2.imshow(img_path, cv2.resize(result, (600,600)))


		cv2.imshow('grid', view)
		cv2.waitKey(0)


def get_pos_and_shape_list(traffic_scene_selected, traffic_scenes_path, labels_path, img_shape):
	info_list = []
	if traffic_scenes_path.endswith(os.sep):
		traffic_scenes_path = traffic_scenes_path[:-1]
	frame = os.path.basename(traffic_scene_selected)
	labels_file_path = os.path.join(labels_path, frame.replace('.png', '.txt'))
	#print(labels_path)
	if os.path.isfile(labels_file_path):
		with open(labels_file_path, 'r') as labels:
			for label in labels:
				label_data = label.split()
				cls = label_data[0]
				x = int(img_shape[1] * float(label_data[1]))
				y = int(img_shape[0] * float(label_data[2]))
				w = int(img_shape[1] * float(label_data[3]))
				h = int(img_shape[0] * float(label_data[4]))
				if w > 0 and h > 0:
					info_list.append({'class': cls, 'pos': int(x + y * img_shape[1]), '(w,h)': (int(w), int(h))})
	return info_list

def generate_sample(idx, targets_path, img_name, traffic_scenes, traffic_scenes_path, labels_path, images_out_path, nb_imgs_generated, binary_annotation_lines_queue, multiclass_annotation_lines_queue, data_out_path, load_path, do_resize=False, use_normal_dist=False):
	#start = time.time()
	np.random.seed(idx)
	img_path = os.path.join(targets_path, img_name)
	try:
		if do_resize:
			target = cv2.resize(cv2.imread(img_path).astype(np.float32), (0,0), fx=2, fy=2) / 255.0
		else:
			target = cv2.imread(img_path).astype(np.float32) / 255.0
	except:
		print(img_path)
		return
	target, add_value, multiply_value = augment_target(target)
	image_data = {
			'bbox_data': []
	}
	bboxes = []
	image_data['add_value'] = add_value
	image_data['multiply_value'] = multiply_value
	if load_path is None:
		scale = None
		tmpt_blur_factor = float(np.random.uniform(0, 3))
		traffic_scene_selected = np.random.choice(traffic_scenes, size=1, replace=False)[0]
		image_data['traffic_scene'] = traffic_scene_selected
		info_list = get_pos_and_shape_list(traffic_scene_selected, traffic_scenes_path, labels_path, target.shape) #np.arange(0, probabilities_vector.size)
		if (len(info_list) > 0):
			probabilities_vector = np.ones(1)
			
			target, bboxes, data = process_img(
				target.copy(), traffic_scene_selected, False, tmpt_blur_factor, probabilities_vector, info_list, add_value, multiply_value, use_normal_dist, 17, 156, 150, 50, bboxes=bboxes, scale=scale)

			for bbox in bboxes:
				image_data['bbox_data'].append({
					'bbox':bbox,
					'data': data
				})
		blur_value = float(np.random.uniform(0, 3))
		image_data['blur_value'] = blur_value
	else:
		with open(os.path.join(load_path, "{:05d}.pkl".format(nb_imgs_generated)),"rb") as data_in_f:
			image_data = pickle.load(data_in_f)

		target, _, _ = augment_target(target, multiply_value=image_data['multiply_value'], add_value=image_data['add_value'])
		bboxes = []
		scale = None
		template_ids_selected = image_data['template_ids']
		for total, bbox_data in enumerate(image_data['bbox_data']):
			template = templates[template_ids_selected[total]][0]
			template_mask = templates[template_ids_selected[total]][1]
			template_category = templates[template_ids_selected[total]][2]

			data = bbox_data['data']
			target, _, _, _ = process_img(target.copy(), template.copy(), template_mask, probabilities_vector, positions_list, image_data['multiply_value'], bboxes=bboxes, scale=scale, prelodaded_data=data)
			bboxes.append(bbox_data['bbox'])

		blur_value = image_data['blur_value']
	t0 = time.time()
	blur_effect = iaa.Sequential([iaa.GaussianBlur(blur_value, deterministic=True)])
	target = blur_effect.augment_image(target)



	img_out_path = os.path.join(images_out_path, "{:05d}_{}.jpg".format(nb_imgs_generated,
																	os.path.splitext(img_name)[0]))
	t0 = time.time()
	cv2.imwrite(img_out_path, (target * 255).astype(np.uint8))
	# print("write time {:.2f}".format(time.time() - t0))
	binary_annotation_lines = []
	multiclass_annotation_lines = []
	for bbox in bboxes:
		if bbox['class'] in ['red', 'green', 'yellow']:
			binary_line = "{},{},{},{},{},{}".format(img_out_path, bbox['xmin'], bbox[
				'ymin'], bbox['xmax'], bbox['ymax'], bbox['class'])
			multiclass_line = "{},{},{},{},{},{}".format(
				img_out_path, bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax'], bbox['category'])

			binary_annotation_lines.append(binary_line)
			multiclass_annotation_lines.append(multiclass_line)

	multiclass_annotation_lines_queue.put(multiclass_annotation_lines)
	binary_annotation_lines_queue.put(binary_annotation_lines)
	t0 = time.time()
	with open(os.path.join(data_out_path, "{:05d}.pkl".format(nb_imgs_generated)),"wb") as data_out_f:
		# json.dump(image_data, data_out_f)
		pickle.dump(image_data, data_out_f)
	# print("dump time {:.2f}".format(time.time() - t0))

def get_traffic_scenes(traffic_scenes_names, traffic_scenes_path):
	traffic_scenes = []
	for traffic_scene_name in traffic_scenes_names:
		traffic_scene_path = os.path.join(traffic_scenes_path, traffic_scene_name)
		traffic_scenes.append(traffic_scene_path)
	return traffic_scenes

if __name__ == '__main__':
	print('[WARN] Before proceeding, check if brightness augmentation is correct!')
	np.random.seed(50)
	
	args = parse_args()
	targets_path = args.targets_path
	traffic_path = args.traffic_path
	labels_path = args.labels_path

	images_out_path = os.path.join(args.out_path, "imgs")
	data_out_path = os.path.join(args.out_path, "data")
	os.makedirs(args.out_path, exist_ok=True)
	os.makedirs(images_out_path, exist_ok=True)
	os.makedirs(data_out_path, exist_ok=True)

	shutil.copyfile(os.path.realpath(__file__), os.path.join(args.out_path, 'generate_dataset.py'))

	all_img_names = os.listdir(targets_path)

	#nb_classes = len(ts_names)
	img_names = np.random.choice(all_img_names, size=(args.total_images))

	t0 = time.time()
	
	traffic_scenes_names = os.listdir(traffic_path)
	traffic_scenes = get_traffic_scenes(traffic_scenes_names, traffic_path)

	MAX_PROCESS = 70
	processes = []
	load_path = args.random_data
	binary_annotation_lines_queue = mp.Queue()
	multiclass_annotation_lines_queue = mp.Queue()
	binary_annotation_lines = []
	multiclass_annotation_lines = []
	nb_imgs_generated = 0
	t0_temp = time.time()
	pbar = tqdm(total=args.total_images)
	for idx, img_name in enumerate(img_names):
			p = mp.Process(target=generate_sample,args=((idx, targets_path, img_name, traffic_scenes, traffic_path, labels_path, images_out_path, idx, binary_annotation_lines_queue, multiclass_annotation_lines_queue, data_out_path, load_path, args.do_resize, args.use_normal_dist)))
			p.daemon = True
			p.start()
			processes.append(p)
			if len(processes) == MAX_PROCESS:
				for p in processes:
					p.join(2)
					pbar.update(1)
					binary_annotation_lines += binary_annotation_lines_queue.get()
					multiclass_annotation_lines += multiclass_annotation_lines_queue.get()
					nb_imgs_generated += 1
					# if nb_imgs_generated % MAX_PROCESS == 0:
						# print("{}/{}".format(nb_imgs_generated, args.total_images))
					if nb_imgs_generated % 1000 == 0:
						t1 = time.time()
						# print("Time spent in last 1000 images: {:.2f}s".format(t1 - t0_temp))
						t0_temp = t1
				processes = []
	pbar.close()


	if len(processes) > 0:
		for p in processes:
			p.join()
			binary_annotation_lines +=binary_annotation_lines_queue.get()
			multiclass_annotation_lines += multiclass_annotation_lines_queue.get()
			nb_imgs_generated += 1
			if nb_imgs_generated % MAX_PROCESS == 0:
				print("{}/{}".format(nb_imgs_generated, args.total_images))
			if nb_imgs_generated % 1000 == 0:
				t1 = time.time()
				print("Time spent in last 1000 images: {:.2f}s".format(t1 - t0_temp))
				t0_temp = t1
		processes = []



	print("Total time: {:.2f}s".format(time.time() - t0))
	print("Generating annotations...")
	# multiclass_annotation_lines = [multiclass_annotation_lines.get() for item in range(amount_per_class*nb_classes)]
	# binary_annotation_lines = [binary_annotation_lines.get() for item in range(amount_per_class*nb_classes)]
	multiclass_annotation_lines
	with open(os.path.join(args.out_path, "multiclass.csv"), "w") as multi_f:
		multi_f.write("\n".join(multiclass_annotation_lines) + "\n")

	with open(os.path.join(args.out_path, "binary.csv"), "w") as binary_f:
		binary_f.write("\n".join(binary_annotation_lines) + "\n")
