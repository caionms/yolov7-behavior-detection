import torch
from torchvision import transforms

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, bbox_iou
from utils.plots import output_to_keypoint, plot_skeleton_kpts

#import matplotlib.pyplot as plt
import cv2
from google.colab.patches import cv2_imshow
import numpy as np

#Listar arquivos
from os import listdir
from os.path import isfile, join

import time

def run_inference(image, model, device):
    # Resize and pad image
    #image = letterbox(image, 960, stride=64, auto=True)[0] # shape: (567, 960, 3)
    # Apply transforms
    #image = transforms.ToTensor()(image) # torch.Size([3, 567, 960])
    #if torch.cuda.is_available():
    #  image = image.half().to(device)
    # Turn image into batch
    #image = image.unsqueeze(0) # torch.Size([1, 3, 567, 960])
    with torch.no_grad():
      output, _ = model(image)
    return output, image

def draw_keypoints(output, image, model, only_keypoints):
  output = non_max_suppression_kpt(output, 
                                     0.25, # Confidence Threshold
                                     0.65, # IoU Threshold
                                     nc=model.yaml['nc'], # Number of Classes
                                     nkpt=model.yaml['nkpt'], # Number of Keypoints
                                     kpt_label=True)
  with torch.no_grad():
        output = output_to_keypoint(output)
  nimg = image[0].permute(1, 2, 0) * 255
  nimg = nimg.cpu().numpy().astype(np.uint8)
  nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

  if(only_keypoints):
    nimg[:] = (0, 0, 0)

  for idx in range(output.shape[0]):
      #plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
      plot_skeleton_kpts_v2(nimg, output[idx, 7:].T, 3)

  return nimg

def plot_skeleton_kpts_v2(im, kpts, steps, box, vehicles_boxes, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    #Conexões entre keypoints
    skeleton = [
                #Rosto 
                [1, 2], [1, 3], [2, 3], [2, 4], [3, 5], [4, 6], [5, 7],
                #Braços
                [6, 7], [6, 8], [7, 9],  [8, 10], [9, 11], 
                #Tronco
                [7, 13], [6, 12], [12, 13],
                #Cintura e pernas
                [14, 12], [15, 13], [16, 14], [17, 15]
      ]

    #Cores para as linhas (seguindo a ordem de skeleton)
    pose_limb_color = palette[[16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9]]
    #Cores para keypoints (seguindo a ordem definida)
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    if isinstance(kpts, np.float64):
      kpts = kpts.tolist()
    num_kpts = len(kpts) // steps
    is_suspect = False #Condição para pintar de suspeito
    r, g, b = 0, 0, 255 #RED - Ordem inversa

    #Condição para saber se está próximo de veículo
    for v_box in vehicles_boxes: 
      #print(bbox_iou_vehicle(box, v_box))
      if bbox_iou_vehicle(box, v_box) > 0:
        is_suspect = True
        #chama o is_squat
        #aqui vou acessar a matriz, se o id desse cara não estiver nela, adiciona e inicia o tempo e guarda a pose
        #caso esteja, calcula o tempo comparando com o atual, caso a pose seja False e a atual True, atualiza
        #utiliza a pose para fazer a condição de tempo

    #Calculate if squat
    #if(is_squat_v4(kpts, steps)):
    #  is_suspect = True
    #  plot_text_box(im, int(80), int(80), "Agachado")
    
    #Plot keypoints
    for kid in range(num_kpts):
        if(not is_suspect):
          r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        #print('Keypoint ' + str(kid) + ': x - ' + str(x_coord) + ' / y - ' +  str(y_coord))
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2] # acima de 0.5 para considerar o ponto correto
                if conf < 0.5:
                    continue
            if(kid == 61 or kid == 122 or kid == 113): # pinta de vermelho se é suspeito
              r = g = b = 255
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
            #plot_number(im, int(x_coord), int(y_coord), (int(r), int(g), int(b)), str(kid))

        
    #cv2.circle(im, (int(421.00), int(491.50)), 5, (0, 0, 255), -1)
    #cv2.circle(im, (int(373.50), int(451.00)), 5, (0, 0, 255), -1)
    #cv2.line(im, (int(kpts[(14)*steps]), int(kpts[(14)*steps+1])), (int(373.50), int(451.00)), (0, 0, 255), thickness=2)
    #cv2.line(im, (int(kpts[(13)*steps]), int(kpts[(13)*steps+1])), (int(421.00), int(491.50)), (0, 0, 255), thickness=2)


    #Plot lines
    for sk_id, sk in enumerate(skeleton):
        if(not is_suspect):
          r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)

def xywh2xyxy_personalizado(boxes):
    """
    Converte caixas delimitadoras no formato [x, y, w, h] para [x1, y1, x2, y2].
    
    Args:
        boxes: Lista de caixas delimitadoras no formato [x, y, w, h].
        
    Returns:
        Lista de caixas delimitadoras no formato [x1, y1, x2, y2].
    """
    
    x, y, w, h = boxes[0], boxes[1], boxes[2], boxes[3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return [x1, y1, x2, y2]
  
def xywh2xyxy_personalizado_2(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = w * (x[0] - x[2] / 2) + padw  # top left x
    y[1] = h * (x[1] - x[3] / 2) + padh  # top left y
    y[2] = w * (x[0] + x[2] / 2) + padw  # bottom right x
    y[3] = h * (x[1] + x[3] / 2) + padh  # bottom right y
    return y
  
def bbox_iou_vehicle(box1, box2):
    """
    Calcula o Índice de sobreposição de Jaccard (IoU) entre duas caixas delimitadoras.

    Parâmetros:
    box1: list[float]
        Lista contendo as coordenadas [x1, y1, x2, y2] da primeira caixa delimitadora.
    box2: list[float]
        Lista contendo as coordenadas [x1, y1, x2, y2] da segunda caixa delimitadora.

    Retorna:
    float
        O valor do Índice de sobreposição de Jaccard (IoU) entre as duas caixas delimitadoras.
    """
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2

    # Coordenadas da intersecção
    x1_intersection = max(x1_box1, x1_box2)
    y1_intersection = max(y1_box1, y1_box2)
    x2_intersection = min(x2_box1, x2_box2)
    y2_intersection = min(y2_box1, y2_box2)

    # Área da intersecção
    intersection_area = max(0, x2_intersection - x1_intersection + 1) * max(0, y2_intersection - y1_intersection + 1)

    # Áreas das caixas delimitadoras
    box1_area = (x2_box1 - x1_box1 + 1) * (y2_box1 - y1_box1 + 1)
    box2_area = (x2_box2 - x1_box2 + 1) * (y2_box2 - y1_box2 + 1)

    # União das áreas
    union_area = box1_area + box2_area - intersection_area

    # Cálculo do IoU
    iou = intersection_area / union_area

    return iou
  
def scale_coords_kpts(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    
    coords[0] -= pad[0]  # x padding
    coords[2] -= pad[0]  # x padding
    coords[1] -= pad[1]  # y padding
    coords[3] -= pad[1]  # y padding
    
    for i in range(len(coords)):
        coords[i] /= gain
    tensor = clip_coords_kpts(coords, img0_shape)
    return tensor.detach().numpy()


def clip_coords_kpts(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    np_array = np.array(boxes)
    tensor = torch.from_numpy(np_array)
    tensor[0].clamp_(0, img_shape[1])  # x1
    tensor[1].clamp_(0, img_shape[0])  # y1
    tensor[2].clamp_(0, img_shape[1])  # x2
    tensor[3].clamp_(0, img_shape[0])  # y2
    return tensor
  
def scale_keypoints_kpts(img1_shape, keypoints, img0_shape, ratio_pad=None):
    # Rescale coords of keypoints (xy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    
    for i in range(17):
      keypoints[(3*i)] -= pad[0]  # x padding
      keypoints[(3*i)+1] -= pad[1]  # y padding
    
    for i in range(17):
      keypoints[(3*i)] /= gain  # x padding
      keypoints[(3*i)+1] /= gain  # y padding

    tensor = clip_keypoints_kpts(keypoints, img0_shape)
    return tensor.detach().numpy()
  
def clip_keypoints_kpts(keypoints, img_shape):
    # Clip bounding xy keypoints to image shape (height, width)
    np_array = np.array(keypoints)
    tensor = torch.from_numpy(np_array)
    for i in range(17):
      tensor[(3*i)].clamp_(0, img_shape[1])  # x
      tensor[(3*i)+1].clamp_(0, img_shape[0])  # y padding
    return tensor