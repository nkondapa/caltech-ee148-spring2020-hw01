import visualize

# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# set a path for saving predictions:
preds_path = './q5q6_preds'

visualize.visualize_all_images_with_bounding_boxes(image_base_path=data_path, prediction_path=preds_path + '/preds.json')
