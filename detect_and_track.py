import os
import cv2
import time
import torch
import argparse
from pathlib import Path
from numpy import random
from random import randint
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path, xywh2xyxy
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, \
                time_synchronized, TracedModel
from utils.download_weights import download

#For keypoint detection
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, bbox_iou
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt
from utils.kpts_utils import run_inference, draw_keypoints, plot_skeleton_kpts_v2, xywh2xyxy_personalizado, xywh2xyxy_personalizado_2

#For SORT tracking
import skimage
from sort import *

#............................... Bounding Boxes Drawing ............................
"""Function to Draw Bounding boxes"""
def draw_boxes_kpts(img, bbox, vehicles_objs, identities=None, categories=None, dic=None, indices_kpts=None, names=None, save_with_object_id=False, path=None,offset=(0, 0)):  
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        plot_skeleton_kpts_v2(img, dic[indices_kpts[i]-1], 3, [x1,y1,x2,y2], vehicles_objs)

        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = str(id) + ":"+ names[cat]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,20), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, [255, 255, 255], 1)
        # cv2.circle(img, data, 6, color,-1)   #centroid of box
        
        txt_str = ""
        if save_with_object_id:
            txt_str += "%i %i %f %f %f %f %f %f" % (
                id, cat, int(box[0])/img.shape[1], int(box[1])/img.shape[0] , int(box[2])/img.shape[1], int(box[3])/img.shape[0] ,int(box[0] + (box[2] * 0.5))/img.shape[1] ,
                int(box[1] + (
                    box[3]* 0.5))/img.shape[0])
            txt_str += "\n"
            with open(path + '.txt', 'a') as f:
                f.write(txt_str)
    return img

"""Function to Draw Bounding boxes for Vehicles"""
#np.array([x1, y1, x2, y2, conf, detclass])
def draw_boxes_vehicles(img, dets_vehicles, save_with_object_id=False, path=None,offset=(0, 0)):
    for x1, y1, x2, y2, conf, detclass in (dets_vehicles):
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        label = "vehicle"

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,20), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, [255, 255, 255], 1)
        # cv2.circle(img, data, 6, color,-1)   #centroid of box
    return img
#..............................................................................


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace, colored_trk, save_bbox_dim, save_with_object_id= opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.colored_trk, opt.save_bbox_dim, opt.save_with_object_id
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))


    #Keypoint detection
    frame_count = 0  #count no of frames
    total_fps = 0  #count total fps
    time_list = []   #list to store time
    fps_list = []    #list to store fps

    #.... Initialize SORT .... 
    #......................... 
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh)
    #......................... 
    
    
    #........Rand Color for every trk.......
    rand_color_list = []
    for i in range(0,5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    #......................................
   

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt or save_with_object_id else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load detection model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Load keypoint model
    model_kpts = attempt_load("yolov7-w6-pose.pt", map_location=device)  #Load model
    _ = model_kpts.eval()
    names = model_kpts.module.names if hasattr(model_kpts, 'module') else model_kpts.names  # get class names

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    
    for path, img, im0s, vid_cap in dataset:
        #kpts_img = img.clone()
        kpts_img = np.copy(img)
        kpts_img = cv2.cvtColor(kpts_img, cv2.COLOR_BGR2RGB)
        kpts_img = letterbox(kpts_img, 960, stride=64, auto=True)[0]
        kpts_img = transforms.ToTensor()(kpts_img) # torch.Size([3, 567, 960])
        if torch.cuda.is_available():
          kpts_img = kpts_img.half().to(device)
        # Turn image into batch
        kpts_img = kpts_img.unsqueeze(0) # torch.Size([1, 3, 567, 960])

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify: #pred = output
            pred = apply_classifier(pred, modelc, img, im0s)

        # Saves the boxes of vehicles
        vehicles_objs = []

        # Saves the boxes of people
        person_objs = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                #..................USE TRACK FUNCTION....................
                #pass an empty array to sort
                dets_vehicles = np.empty((0,6))

                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    if(detclass == 2 or detclass == 3): #adiciona os veiculos na lista
                        dets_vehicles = np.vstack((dets_vehicles, 
                                np.array([x1, y1, x2, y2, conf, detclass])))
                        vehicles_objs.append([x1,y1,x2,y2])
                    #elif(detclass == 0): #adiciona pessoas na lista remover
                    #    person_objs.append([x1,y1,x2,y2])
                    #    dets_to_sort = np.vstack((dets_to_sort, 
                    #            np.array([x1, y1, x2, y2, conf, detclass])))
                        
                # chamar o método de output to keypoint e com o resultado fazer outro for, esse for irei chamar o dets_to_sort
                
                #dicionario auxiliar para keypoints
                dic = {}

                # Chama o output_to_keypoint (dentro do draw) para detectar os keypoints
                #vid_cap = cv2.cvtColor(vid_cap, cv2.COLOR_BGR2RGB)
                output, img = run_inference(kpts_img, model_kpts, device)
                output = non_max_suppression_kpt(output, 
                                     0.25, # Confidence Threshold
                                     0.65, # IoU Threshold
                                     nc=model_kpts.yaml['nc'], # Number of Classes
                                     nkpt=model_kpts.yaml['nkpt'], # Number of Keypoints
                                     kpt_label=True)
                with torch.no_grad():
                        output = output_to_keypoint(output)
                        
                #pass an empty array to sort
                dets_to_sort = np.empty((0,7))

                #batch_id, class_id, x, y, w, h, conf, *kpts
                for idx in range(output.shape[0]):
                  batch_id = output[idx, 0]
                  class_id = output[idx, 1]
                  x = output[idx, 2]
                  y = output[idx, 3]
                  w = output[idx, 4]
                  h = output[idx, 5]
                  print('x y w h')
                  print(x, y, w, h)
                  conf = output[idx, 6]
                  keypoints = output[idx, 7:].T

                  if(class_id == 0): #chama o tracking para pessoas
                    x1,y1,x2,y2 = xywh2xyxy_personalizado([x, y, w, h])
                    dic[idx] = keypoints
                    #print('vetor sendo salvo no tracker')
                    #print(x1, y1, x2, y2, conf, class_id, idx)
                    #w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    #h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    dets_to_sort = np.vstack((dets_to_sort, 
                            np.array([x1, y1, x2, y2, conf, class_id, idx])))

                #print('dets to sort')
                #print(dets_to_sort)
                        
                # Run SORT
                tracked_dets = sort_tracker.update_kpts(dets_to_sort)
                tracks = sort_tracker.getTrackers()

                txt_str = ""
                
                #loop over tracks
                for track in tracks:
                    # color = compute_color_for_labels(id)
                    '''
                    #draw colored tracks
                    if colored_trk:
                        [cv2.line(im0, (int(track.centroidarr[i][0]),
                                    int(track.centroidarr[i][1])), 
                                    (int(track.centroidarr[i+1][0]),
                                    int(track.centroidarr[i+1][1])),
                                    rand_color_list[track.id], thickness=2) 
                                    for i,_ in  enumerate(track.centroidarr) 
                                      if i < len(track.centroidarr)-1 ] 
                    #draw same color tracks
                    else:
                        [cv2.line(im0, (int(track.centroidarr[i][0]),
                                    int(track.centroidarr[i][1])), 
                                    (int(track.centroidarr[i+1][0]),
                                    int(track.centroidarr[i+1][1])),
                                    (255,0,0), thickness=2) 
                                    for i,_ in  enumerate(track.centroidarr) 
                                      if i < len(track.centroidarr)-1 ] 
                    '''
                    if save_txt and not save_with_object_id:
                        # Normalize coordinates
                        txt_str += "%i %i %f %f" % (track.id, track.detclass, track.centroidarr[-1][0] / im0.shape[1], track.centroidarr[-1][1] / im0.shape[0])
                        if save_bbox_dim:
                            txt_str += " %f %f" % (np.abs(track.bbox_history[-1][0] - track.bbox_history[-1][2]) / im0.shape[0], np.abs(track.bbox_history[-1][1] - track.bbox_history[-1][3]) / im0.shape[1])
                        txt_str += "\n"
               
                        
                if save_txt and not save_with_object_id:
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(txt_str)
                                
                #só vai ter pessoas, então posso fazer a chamada passando como um array, a lista de kpts que nem categories (da para pegar no for do output_to_keypoints)
                #dentro do método de draw_boxes chamar o plot skeleton
                # draw boxes for visualization 
                
                #print('tracked dets')
                #print(tracked_dets)

                if len(tracked_dets)>0:
                    bbox_xyxy = tracked_dets[:,:4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    indices_kpts = tracked_dets[:, 8]
                    #dentro do draw_boxes, testar se intercepta (passando a lista de veiculos)
                    draw_boxes_kpts(im0, bbox_xyxy, vehicles_objs, identities, categories, dic, indices_kpts, names, save_with_object_id, txt_path)

                draw_boxes_vehicles(im0, dets_vehicles, save_with_object_id, txt_path)  
                #........................................................
                
                #fazer um for que passe pela lista de veiculos que criei e chama o plot_box ou draw_box mostrando vehicle
                
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            
        
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                  cv2.destroyAllWindows()
                  raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img or save_with_object_id:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

def keypoint_detection(img, model_kpts, device):  
    img = img.to(device)  #convert image data to device
    img = img.float() #convert image to float precision (cpu)
    start_time = time.time() #start time for fps calculation

    with torch.no_grad():  #get predictions
        output_data, _ = model_kpts(img)

    output_data = non_max_suppression_kpt(output_data,   #Apply non max suppression
                                0.25,   # Conf. Threshold.
                                0.65, # IoU Threshold.
                                nc=model_kpts.yaml['nc'], # Number of classes.
                                nkpt=model_kpts.yaml['nkpt'], # Number of keypoints.
                                kpt_label=True)

    output = output_to_keypoint(output_data)

    im0 = img[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
    im0 = im0.cpu().numpy().astype(np.uint8)
    
    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

    for i, pose in enumerate(output_data):  # detections per image
    
        if len(output_data):  #check if no pose
            for c in pose[:, 5].unique(): # Print results
                n = (pose[:, 5] == c).sum()  # detections per class
                print("No of Objects in Current Frame : {}".format(n))
            
            for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
                c = int(cls)  # integer class
                kpts = pose[det_index, 6:]
                plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True), 
                            line_thickness=opt.line_thickness,kpt_label=True, kpts=kpts, steps=3, 
                            orig_shape=im0.shape[:2])
        return im0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--download', action='store_true', help='download model weights automatically')
    parser.add_argument('--no-download', dest='download', action='store_false',help='not download model weights if already exist')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='object_tracking', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--colored-trk', action='store_true', help='assign different color to every track')
    parser.add_argument('--save-bbox-dim', action='store_true', help='save bounding box dimensions with --save-txt tracks')
    parser.add_argument('--save-with-object-id', action='store_true', help='save results with object id to *.txt')

    parser.set_defaults(download=True)
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    if opt.download and not os.path.exists(''.join(opt.weights)):
        print('Model weights not found. Attempting to download now...')
        download('./')

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
