import environment as env
import yaml
import os
import cv2
import numpy as np
from PIL import Image

project_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
yaml_path = project_path + "/args.yaml"

with open(yaml_path) as f:
    yaml_file = yaml.load(f)

map_path = project_path + "/" + yaml_file['map_name']
visible_threshold = yaml_file['visible_threshold']
n_angle = yaml_file['n_angle']
step_size = yaml_file['step_size']

game = env.Environment(map_path, visible_threshold, n_angle, step_size)

game.start()

# a = game.getImage()
# for i in range(20):
#     for j in range(20):
#         print(a[i][j])
while(True):
    img = Image.fromarray(np.uint8(game.getImage()), 'L')
    img = img.resize((300,300))
    img.show()
    
    action = input()
    if action == -1:
        break
    reward, success, done = game.doAction(action)
    print(reward)
    print(success)
    print(done)
    
#grayImage = cv2.cvtColor(np.array(game.getImage()), cv2.COLOR_GRAY2BGR)