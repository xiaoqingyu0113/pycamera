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
import json
import os 
import sys
import csv

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


def draw_lines_on_image(image, points, idx=0):
    colormap = mlp.colormaps['viridis']
    color = colormap(idx/7.0)[:3]  # Get the RGBA values from the colormap
    color = tuple([int(255* color[2]), int(255*color[1]), int(255*color[0])]) # BGR
    for i in range(1, len(points)):
        cv2.line(image, (int(points[i-1][0]), int(points[i-1][1])), (int(points[i][0]), int(points[i][1])), color, 1)

def draw_points_on_image(image, points, idx=0):
    colormap = mlp.colormaps['viridis']
    color = colormap(idx/7.0)[:3]  # Get the RGBA values from the colormap
    color = tuple([int(255* color[2]), int(255*color[1]), int(255*color[0])]) # BGR
    for pt in points:
        cv2.circle(image, (int(pt[0]), int(pt[1])), 3, color, -1)
    return image

def add_detection(separated_detections, camera,msg, yolo_det, start_point):
    name, prob, (x, y, w, h) = yolo_det
    thresh = 200

    
    # - If no detection in the list, add the first one            
    # - Also make sure the added point is close to the start point
    if len(separated_detections[camera]) == 0:
        dist = np.linalg.norm(np.array([x,y]) - np.array(start_point))
        if  dist < thresh:
            separated_detections[camera].append([[x,y, msg.header.stamp.secs, msg.header.stamp.nsecs, camera]])
        return separated_detections

    # - If new ball launched during the launch, add a new trajectory
    dist = np.linalg.norm(np.array([x,y]) - np.array(start_point))

    if len(separated_detections[camera][-1]) > 60 and dist < 70:
        separated_detections[camera].append([[x,y, msg.header.stamp.secs, msg.header.stamp.nsecs, camera]])
        return separated_detections
    
    # - If the detection is close to the last point of the trajectory, add it to the trajectory
    # - We might prioritize the lastest trajectory before checking previous trajectory
    else:
        curr_points = np.array([x,y])      
        added_points = np.array([[det[0], det[1]] for det in separated_detections[camera][-1]])   
        
        dists = np.linalg.norm(curr_points - added_points, axis=1)
        dist_min = min(dists)       
        dist_last = min(dists[-4:]) if len(dists) > 4 else dists[-1] 


        # check the latest trajectory
        if dist_last < thresh and dist_min > 5.0:
            separated_detections[camera][-1].append([x,y, msg.header.stamp.secs, msg.header.stamp.nsecs, camera])
            return separated_detections
        # check previous trajectories
        else:
            close_idx, close_dist = None, 1000
            for traj_idx, traj in enumerate(separated_detections[camera]):
                curr_points = np.array([x,y])      
                added_points = np.array([[det[0], det[1]] for det in traj])   
                dists = np.linalg.norm(curr_points - added_points, axis=1)
                dist_last = min(dists[-4:]) if len(dists) > 4 else dists[-1] 
                dist_min = min(dists)    
                if dist_last < close_dist and dist_min > 5.0:
                    close_dist = dist_last
                    close_idx = traj_idx
            if close_dist < thresh:
                separated_detections[camera][close_idx].append([x,y, msg.header.stamp.secs, msg.header.stamp.nsecs, camera])
            return separated_detections

def check_local_yolo_on_video(bag_reader, yolo_setups, camera_topic,
                               is_imshow=True, 
                               is_save_video = False,
                               is_save_detections = False,
                               is_update_summary = False,
                               start_points = {'camera_1': [1150, 655], 'camera_2': [278, 650], 'camera_3': [210, 350]}):
    '''
    generate debug video
    '''
    fourcc = cv2.VideoWriter_fourcc(*'VP90') 
    filename = bag_reader.filename.split('/')[-1].split('.')[0]

    if is_save_video:
        print('generating debug video...')
        vws = {f"camera_{i+1}": cv2.VideoWriter(f'data/real/debug/detections/video/camera_{i+1}_{filename}.webm', fourcc, 20, (1280, 1024)) for i in range(3)}

    # setup yolo
    network, class_names, darknet_image = yolo_setups

    prev_image_by_camera = {f"camera_{i+1}": None for i in range(3)}
    separated_detections = {f"camera_{i+1}": [] for i in range(3)}
    spinner = ['|', '/', '-', '\\']

    # assume image messages come before detection messages
    for count, (topic, msg, t) in enumerate(bag_reader.read_messages()): 
        if 'image_color' in topic and camera_topic in topic:

            # read image
            img_data = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

            img_data_copy = img_data.copy()

            # curr_gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
            # if prev_image_by_camera[camera] is not None:
            #     diff = (curr_gray - prev_image_by_camera[camera]) < 30
            #     img_data_copy[diff] = [0, 0, 0]

            camera = msg.header.frame_id

            darknet.copy_image_from_bytes(darknet_image, img_data_copy.tobytes())
            yolo_detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.5)

            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            for yolo_det in yolo_detections:
                # hate this, but don't want to discard this data
                if 'p4_p1' in filename:
                    p4_p1_start_points = {'camera_1': [954, 460], 'camera_2': [519, 463], 'camera_3': [210, 350]}
                    separated_detections = add_detection(separated_detections, camera, msg, yolo_det, p4_p1_start_points[camera])
                else:
                    separated_detections = add_detection(separated_detections, camera, msg, yolo_det, start_points[camera])
                if is_imshow:
                    name, prob, (x, y, w, h) = yolo_det
                    cv2.circle(img_data, (int(x), int(y)), 5, (255, 0, 0), -1)
                    cv2.putText(img_data, name, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            

            
            # draw frames
            if is_save_video or is_imshow:
                for traj_idx, traj in enumerate(separated_detections[camera]):
                    img_data = draw_points_on_image(img_data, traj, idx=traj_idx)

                # if traj_idx defined
                if 'traj_idx' in locals():
                    x, y = separated_detections[camera][traj_idx][-1][:2]
                    cv2.putText(img_data, f'{traj_idx}', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
                cv2.putText(img_data, f'count={count}', (1280//2, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

            if is_save_video:
                vws[camera].write(img_data)
            if is_imshow:
                cv2.imshow('yolo', img_data)
                cv2.waitKey(0)

            # show spinner
            sys.stdout.write(f"\rProcessing frame {count}... {spinner[count % len(spinner)]}")
            sys.stdout.flush()


    # delete trajectory if detection is less than 10 points
    for camera, detections in separated_detections.items():
        separated_detections[camera] = [traj for traj in detections if len(traj) >= 10]
    # conclude the results
    print('\n')
    for camera, detections in separated_detections.items():
        print(f'{camera}: {len(detections)} trajectories')

    # save detections
    if is_save_detections:
        with open(f"data/real/detections/{filename}.json", 'w') as f:
            json.dump(separated_detections, f, indent=4)
        print(f'Separated detections saved to data/real/detections/{filename}.json')

    if is_update_summary:
        _update_detection_summary()

    if is_save_video:
        # release video writers
        for vw in vws.values():
            vw.release()
        print(f"Debug video saved to data/real/debug/detections/video/{filename}.webm")
    
    return separated_detections

def yolo_setup():
    config_file = 'conf/darknet/yolov4-lite.cfg'
    data_file = 'conf/darknet/obj.data'
    weights = 'conf/darknet/yolov4-lite_pingpong_final.weights'
    network, class_names, class_colors = darknet.load_network(config_file,data_file,weights)
    darknet_image = darknet.make_image(1280, 1024, 3)
    return network, class_names, darknet_image

def extract_single_bag(bagfile, yolo_setups):
    bag_reader = rosbag.Bag(bagfile)
    # show_topic_info(bag_reader)
    # start_points = click_start_point()
    separated_detections = check_local_yolo_on_video(bag_reader, yolo_setups, 'camera', # 'camera' means all cameras
                                is_imshow=False,
                                is_save_detections=True,
                                is_update_summary=True,
                                is_save_video=False)
    bag_reader.close()
    return separated_detections

def _update_detection_summary():
    saved_traj_files = glob.glob('data/real/detections/*.json')
    saved_traj_files = sorted(saved_traj_files)

    summary_list = []
    for traj_file in saved_traj_files:
        with open(traj_file, 'r') as f:
            separated_detections = json.load(f)
        name = traj_file.split('/')[-1].split('.')[0]
        summary_list.append([name])
        for camera, detections in separated_detections.items():
            summary_list[-1].append(len(detections))

    with open('data/real/detections/_summary.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['bagfile', 'camera_1', 'camera_2', 'camera_3'])
        writer.writerows(summary_list)
        print('Summary updated')

def extract_all_bags(bag_dir):
    bag_files = glob.glob(bag_dir + '*.bag')
    yolo_setups = yolo_setup()

    summary_list = []

    def display_summary(summary_list):
        print('-'*80)
        print(' '*5, '\tcamera_1', '\tcamera_2', '\tcamera_3')
        for row in summary_list:
            print(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}")

        # save to csv
        with open('data/real/detections/_summary.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['bagfile', 'camera_1', 'camera_2', 'camera_3'])
            writer.writerows(summary_list)
            print('Summary saved to data/real/detections/_summary.csv')
        

    for bagfile in tqdm.tqdm(bag_files):
        print(f'Processing {bagfile}...')
        separated_detections = extract_single_bag(bagfile, yolo_setups)

        # save summary
        name = bagfile.split('/')[-1].split('.')[0]
        summary_list.append([name])
        for camera, detections in separated_detections.items():
            summary_list[-1].append(len(detections))
        display_summary(summary_list)

def view_single_separated_detections(detection_file, is_imshow=False):
    with open(detection_file, 'r') as f:
        separated_detections = json.load(f)
    
    bg_images = {f'camera_{i+1}': cv2.imread(f'data/real/bg/camera_{i+1}.jpg') for i in range(3)}
    for camera, detections in separated_detections.items():
        for idx, traj in enumerate(detections):
            draw_lines_on_image(bg_images[camera], traj, idx)
            draw_points_on_image(bg_images[camera], traj, idx)
    
    for camera, img in bg_images.items():
        cv2.imwrite(f'data/real/debug/detections/separate/{camera}_{detection_file.split("/")[-1].split(".")[0]}_sep_det_local.jpg', img)

    if is_imshow:
        for camera, img in bg_images.items():
            cv2.imshow(camera, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def view_all_separated_detections(detection_dir):
    import os
    detection_files = glob.glob(detection_dir + '*.json')
    for detection_file in detection_files:
        if 'clean' not in detection_file:
            view_single_separated_detections(detection_file)


class DetectionEditor:
    def __init__(self):
        # load background images (generated by 'extract_and_save_background_images')
        self.bg_images = {f'camera_{i+1}': cv2.imread(f'data/real/bg/camera_{i+1}.jpg') for i in range(3)}

        self.prev_operation = []
        self.choice = None

    def read_detections(self, detection_file):
        # load separated detections (generated by 'check_local_yolo_on_video', or 'extract_single_bag')
        with open(detection_file, 'r') as f:
            separated_detections = json.load(f)
        return separated_detections
    
    def draw_all_detections(self, bg_image, detections, line_style='point+line'):
        if line_style == 'line':
            for idx, traj in enumerate(detections):
                draw_lines_on_image(bg_image, traj, idx)
        elif line_style == 'point':
            for idx, traj in enumerate(detections):
                draw_points_on_image(bg_image, traj, idx)
        elif line_style == 'point+line':
            for idx, traj in enumerate(detections):
                draw_lines_on_image(bg_image, traj, idx)
                draw_points_on_image(bg_image, traj, idx)
        return bg_image
    
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            camera, = param
            new_bg_image = self.bg_images[camera].copy()
            tid, did = self._find_closest_index_and_detection(self.detections, x, y)
            self.choice = (tid, did)
            det = self.detections[tid][did]

            width = 10//2
            self.draw_all_detections(new_bg_image, self.detections)
            cv2.rectangle(new_bg_image, (int(det[0]-width), int(det[1]-width)), (int(det[0]+width), int(det[1]+width)), (0, 0, 255), 1)
            cv2.circle(new_bg_image, (int(det[0]), int(det[1])), 2, (0, 0, 255), -1)
            cv2.imshow(self.detection_file, new_bg_image)

    def _find_closest_index_and_detection(self, detections, x, y):
        curr_click = np.array([x, y])
        min_dist, min_tid, min_did = 1000, None, None
        for tid, traj in enumerate(detections):
            for did, det in enumerate(traj):
                dist = np.linalg.norm(curr_click - np.array(det[:2]))
                if dist < min_dist:
                    min_dist = dist
                    min_tid = tid
                    min_did = did
        return min_tid, min_did

    def _delete_detection(self, detections, tid, did):
        if len(detections[tid]) > 0:
            det = detections[tid].pop(did)
            self.prev_operation.append(['d', [tid, did, det]])
        return detections
    
    def _delete_to_end(self, detections, tid, did):
        if len(detections[tid]) > 0:
            self.prev_operation.append(['e', [tid, did, detections[tid][did:]]])
            detections[tid] = detections[tid][:did]
        return detections
    
    def _undo_operation(self, detections):
        if len(self.prev_operation) > 0:
            key, (tid, did, data) = self.prev_operation.pop()
            if key == 'e':
                detections[tid].extend(data)
                print('undo delete to end')
            elif key == 'd':
                detections[tid].insert(did, data)
                print('undo delete')
        return detections
    
    def _manual_edit_operation(self,camera, detections):
        # initial setup
        self.detections = detections
        bg_image = self.bg_images[camera].copy()
        self.draw_all_detections(bg_image, detections)
        cv2.imshow(self.detection_file, bg_image)
        cv2.setMouseCallback(self.detection_file, self.on_mouse, param=(camera,))

        # edit loop
        while True:
            bg_image = self.bg_images[camera].copy()
            cv2.imshow(self.detection_file, self.draw_all_detections(bg_image, self.detections))
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
            elif key == ord('e'):
                self.detections = self._delete_to_end(self.detections, self.choice[0], self.choice[1])
            elif key == ord('u'):
                self.detections = self._undo_operation(self.detections)
            elif key == ord('d'):
                self.detections = self._delete_detection(self.detections, self.choice[0], self.choice[1])


    def _set_clean_detection_file(self, detection_file):
        if 'clean' in detection_file:
            self.clean_detection_file = detection_file
        else:
            base, ext = os.path.splitext(detection_file)
            self.clean_detection_file = f"{base}_clean{ext}"

    def _run(self, detection_file):
        self.detection_file = detection_file
        self._set_clean_detection_file(detection_file)
        separated_detections = self.read_detections(detection_file)

        for camera, detections in separated_detections.items():
            print(f'{camera}: {len(detections)} trajectories')

        # loop each camera
        for camera, detections in separated_detections.items():
            self._manual_edit_operation(camera, detections)
            cv2.destroyWindow(detection_file)

        # save cleaned detections after loop
        with open( self.clean_detection_file, 'w') as f:
            json.dump(self.detections, f, indent=4)
            print(f'saved to {self.clean_detection_file}')

    def run(self, detection_files):
        if type(detection_files) == str:
            self._run([detection_files])
        else:
            for detection_file in detection_files:
                self._run(detection_file)

    def remove_cleaned_detections(self, detection_file):
        self.detection_file = detection_file
        self._set_clean_detection_file(detection_file)
        if os.path.exists(self.clean_detection_file):
            os.remove(self.clean_detection_file)
            print(f'{self.clean_detection_file} removed')
        else:
            print(f'{self.clean_detection_file} does not exist')
                
    def remove_all_cleaned_detections(self, detection_dir):
        import os
        detection_files = glob.glob(detection_dir + '*.json')
        for detection_file in detection_files:
            if 'clean' in detection_file:
                os.remove(detection_file)
                print(f'{detection_file} removed')

if __name__ == '__main__':
    '''
    yolo raw detection extraction
    '''
    bagfile = '/home/qingyu/bag_files/lfg/p5_p1.bag'
    
    check_local_yolo_on_video(rosbag.Bag(bagfile), yolo_setup(), 
                              camera_topic='camera_2', 
                              is_imshow=True,
                              is_save_detections=False,
                              is_save_video=False)
    # _update_detection_summary()
    # extract_single_bag(bagfile, yolo_setup())      
    # extract_all_bags('/home/qingyu/bag_files/lfg/')

    '''          
    visualize detections
    '''
    # view_single_separated_detections('data/real/detections/p2_n3.json', is_imshow=True)
    # view_all_separated_detections('data/real/detections/')

    '''
    manual edit detections, save with suffix '_clean'. If _clean file exists, use it instead of original file
    '''
    # de = DetectionEditor()
    # de.run('data/real/detections/p4_p3.json')
    # de.run(glob.glob('data/real/detections/*.json'))

