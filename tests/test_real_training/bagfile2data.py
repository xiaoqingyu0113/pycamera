import rosbag
from bagpy import bagreader
import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib as mlp
import tqdm
import glob
from lfg.darknet import darknet

def show_topic_info(bag_reader):
    info = bag_reader.get_type_and_topic_info() 
    topic_value = info.topics.values()
    topics = info.topics.keys()
    print('-'*50)
    print('Bag Infos: ')
    for topic, val in zip(topics, topic_value):
        print(f"TOPIC:{topic} \tTYPE:{val.msg_type}\tCOUNT: {val.message_count}\tCONNECTIONS: {val.connections}\tFREQ: {val.frequency}")
    print('-'*50)


def extract_and_save_background_images(bag_reader):
    visited = []
    for count, (topic, msg, t) in enumerate(bag_reader.read_messages()): 
        if topic not in visited and 'image_color' in topic:
            camera = topic.split('/')[1]
            img_data = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite(f'data/real/bg/{camera}.jpg', img_data)
            visited.append(topic)
            if len(visited) == 3:
                break

def extract_detections(bag_reader,save_debug_images=False):
    '''
    output: Dict[frame_id] = [[x, y, sec, nsec, frame_id, filename], ...]
    '''
    detections = {'camera_1': [], 'camera_2': [], 'camera_3': []}
    file_name = bag_reader.filename.split('/')[-1].split('.')[0]
    # get detections
    for count, (topic, msg, t) in enumerate(bag_reader.read_messages()): 
        if 'detection' in topic:
            for pt in msg.points:
                detections[msg.header.frame_id].append([pt.x, pt.y, msg.header.stamp.secs, msg.header.stamp.nsecs, msg.header.frame_id, file_name])
        # print(topic, msg.header.frame_id, msg.header.stamp.secs, msg.header.stamp.nsecs)
    # plot debug images with raw detections
    if save_debug_images:
        for camera in detections.keys():
            im = cv2.imread(f'data/real/bg/{camera}.jpg')
            draw_points_on_image(im, detections[camera], idx=0)
            cv2.imwrite(f'data/real/debug/detections/raw/{camera}_{file_name}_raw_detections.jpg', im)           
    return detections

def separate_detections(detections, start_points, save_debug_images=False):
    separated_detections = {'camera_1': [], 'camera_2': [], 'camera_3': []}
    for camera in detections.keys():
        separated_detections[camera] = separate_points(detections[camera], start_points[camera])
        
    if save_debug_images:
         for camera in detections.keys():
            im = cv2.imread(f'data/real/bg/{camera}.jpg')
            for idx, track in enumerate(separated_detections[camera]):
                draw_lines_on_image(im, track, idx=idx)
            file_name = bag_reader.filename.split('/')[-1].split('.')[0]
            cv2.imwrite(f'data/real/debug/detections/separate/{camera}_{file_name}_sep_detections.jpg', im)
    return separated_detections

    
def separate_points(points, start_point):

    def check_tail(pt, separated_points):
        if len(separated_points) == 0:
            return None, 1000
        max_dist = 1000
        max_idx = -1
        for idx, track in enumerate(separated_points):
            dist = np.linalg.norm(np.array(pt[:2]) - np.array(track[-1][:2]))
            if dist < max_dist:
                max_dist = dist
                max_idx = idx
        return max_idx, max_dist
    
    separated_points = []
    single_track = []

    # find the first point close to the launcher

    for start_id, pt in enumerate(points):
        if np.linalg.norm(np.array(pt[:2]) - np.array(start_point)) < 30:
            single_track.append(pt)
            break

    # iterate the rest of the points
    for pt in points[start_id+1:]:
        is_new_track = np.linalg.norm(np.array(pt[:2]) - np.array(start_point)) < 30 and len(single_track) > 30
        max_idx, max_dist = check_tail(pt, separated_points)
        curr_dist = np.linalg.norm(np.array(pt[:2]) - np.array(single_track[-1][:2]))
        is_curr_track = curr_dist < 30 and curr_dist < max_dist
        is_prev_track = max_dist < 30 and max_dist < curr_dist
        if is_new_track:
            separated_points.append(single_track)
            single_track = [pt]
        elif is_curr_track:
            single_track.append(pt)
        elif is_prev_track:
            separated_points[max_idx].append(pt)

    separated_points.append(single_track)

    return [sp for sp in separated_points if len(sp) > 80] # filter out short tracks

def draw_points_on_image(image, points, idx=0):
    colormap = mlp.colormaps['viridis']
    color = colormap(idx/7.0)[:3]  # Get the RGBA values from the colormap
    color = tuple(int(c * 255) for c in color)
    for pt in points:
        cv2.circle(image, (int(pt[0]), int(pt[1])), 2, color, -1)
    return image

def draw_lines_on_image(image, points, idx=0):
    colormap = mlp.colormaps['viridis']
    color = colormap(idx/7.0)[:3]  # Get the RGBA values from the colormap
    color = tuple(int(c * 255) for c in color)
    for i in range(1, len(points)):
        cv2.line(image, (int(points[i-1][0]), int(points[i-1][1])), (int(points[i][0]), int(points[i][1])), color, 2)

def click_start_point():
    start_points = {'camera_1': [], 'camera_2': [], 'camera_3': []}
    thred = 150
    def get_start_point(event, x, y, flags, param):
        start_points, img, im = param
        if event == cv2.EVENT_LBUTTONDOWN:
            for camera in start_points.keys():
                if camera in img:
                    start_points[camera] = [x, y]
                    cv2.rectangle(im, (x - thred//2, y - thred//2), (x + thred//2, y + thred//2), (0, 255, 0), 2)
                    cv2.imshow(img, im)
                    break

    debug_images = glob.glob('data/real/debug/detections/raw/*.jpg')
    for img in debug_images:
        im = cv2.imread(img)
        cv2.imshow(img, im)
        cv2.setMouseCallback(img, get_start_point, param=(start_points, img, im))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(start_points)
    return start_points


def check_sep_detect_on_video(detections, bag_reader):
    '''
    generate debug video
    '''
    print('generating debug video...')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    filename = bag_reader.filename.split('/')[-1].split('.')[0]
    vws = {f"camera_{i+1}": cv2.VideoWriter(f'data/real/debug/detections/video/camera_{i+1}_{filename}.avi', fourcc, 20, (1280, 1024)) for i in range(3)}
    sd_idx = {f"camera_{i+1}": 0 for i in range(3)}

    # assume image messages come before detection messages
    for count, (topic, msg, t) in tqdm.tqdm(enumerate(bag_reader.read_messages())): 
        if 'image_color' in topic:
            camera = msg.header.frame_id
            if sd_idx[camera] >= len(detections[camera]):
                    continue
            detection = detections[camera][sd_idx[camera]]
            img_data = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)

            while detection[2] == msg.header.stamp.secs and detection[3] == msg.header.stamp.nsecs and detection[4] == msg.header.frame_id:
                sd_idx[camera] += 1
                if sd_idx[camera] >= len(detections[camera]):
                    break
                detection = detections[camera][sd_idx[camera]]
                
                # draw detection on image
                img_data = cv2.circle(img_data, (int(detection[0]), int(detection[1])), 2, (0, 255, 0), -1)
                # print(img_data.shape)
                vws[camera].write(img_data)

    for k, vw in vws.items():
        vw.release()

    # print(sd_idx)
    # print(count)
    for cm, sp in detections.items():
        print(cm, len(sp))
    
    print('done!')

def check_local_yolo_on_video(bag_reader):
    '''
    generate debug video
    '''
    print('generating debug video...')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    filename = bag_reader.filename.split('/')[-1].split('.')[0]
    vws = {f"camera_{i+1}": cv2.VideoWriter(f'data/real/debug/detections/video/camera_{i+1}_{filename}.avi', fourcc, 20, (1280, 1024)) for i in range(3)}
    
    # setup yolo
    config_file = 'conf/darknet/yolov4-lite.cfg'
    data_file = 'conf/darknet/obj.data'
    weights = 'conf/darknet/yolov4-lite_pingpong_final.weights' # for lab pc
    network, class_names, class_colors = darknet.load_network(config_file,data_file,weights)
    darknet_image = darknet.make_image(1280, 1024, 3)

    prev_image_by_camera = {f"camera_{i+1}": None for i in range(3)}

    # assume image messages come before detection messages
    for count, (topic, msg, t) in tqdm.tqdm(enumerate(bag_reader.read_messages())): 
        if 'image_color' in topic and 'camera_1' in topic:
            # read image
            img_data = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

            if prev_image_by_camera[msg.header.frame_id] is None:
                prev_image_by_camera[msg.header.frame_id] = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY).astype(int)
                continue

            # # remove background
            # diff_thresh = 15
            # gray1 = prev_image_by_camera[msg.header.frame_id]
            # gray2 = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY).astype(int)
            # frame_diff = ((gray1 - gray2) < diff_thresh) & ((gray2 - gray1) < diff_thresh) 
            # prev_image_by_camera[msg.header.frame_id] = gray2
            # img_for_yolo = img_data.copy()
            # img_for_yolo[frame_diff] = 0
            

            darknet.copy_image_from_bytes(darknet_image, img_data.tobytes())
            yolo_detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.5)
            print('detection prob:')
            for yolo_det in yolo_detections:
                name, prob, (x, y, w, h) = yolo_det
                if float(prob) < 80:
                    continue
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)
                img_data = cv2.rectangle(img_data, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(img_data, str(prob)[:2], (int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            cv2.imshow('yolo', img_data)
            cv2.waitKey(0)
            

    for k, vw in vws.items():
        vw.release()

    # print(sd_idx)
    # print(count)
    for cm, sp in detections.items():
        print(cm, len(sp))
    
    print('done!')

if __name__ == '__main__':
    bagfile = '/home/qingyu/bag_files/lfg/n3_n3.bag'
    bag_reader = rosbag.Bag(bagfile)
    show_topic_info(bag_reader)

    # run this for the first time to extract background images
    # extract_and_save_background_images(bag_reader)

    # bag_dir = '/home/qingyu/bag_files/lfg/'

    # detections = extract_detections(bag_reader,save_debug_images=True)

    # run this if you want to manually select the start points
    start_points = click_start_point()
    start_points = {'camera_1': [1236, 679], 'camera_2': [198, 724], 'camera_3': [179, 368]}

    # separated_detections = separate_detections(detections, start_points, save_debug_images=True)
    # check_sep_detect_on_video(detections, bag_reader)

    # check_local_yolo_on_video(bag_reader)

    bag_reader.close()