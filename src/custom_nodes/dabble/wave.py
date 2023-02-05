
"""
Custom node to show keypoints and count the number of times the person's hand is waved
"""
import time, math, random
from typing import Any, Dict, List, Tuple, Iterable, Union
import cv2
import numpy as np
from peekingduck.pipeline.nodes.abstract_node import AbstractNode
start = time.time()
tens = start//10
# setup global constants
FONT = cv2.FONT_HERSHEY_DUPLEX

from peekingduck.pipeline.nodes.draw.utils.constants import (
    CHAMPAGNE,
    POINT_RADIUS,
    THICK,
    TOMATO,
)
from peekingduck.pipeline.nodes.draw.utils.general import (
    get_image_size,
    project_points_onto_original_image,
)

startPose =[[0.51125367, 0.17319976], 
  [0.5215636,  0.1603285 ] ,
  [0.49684895, 0.15743376] ,
  [0.54875122, 0.16919625] ,
  [0.48070042, 0.16239033] ,
  [0.57799855, 0.2312702 ] ,
  [0.44527747, 0.23021722] ,
  [0.66825381, 0.15130381] ,
  [0.35764033, 0.18226056] ,
  [0.73405464, 0.09322681] ,
  [0.28611953, 0.12195434] ,
  [0.54399387, 0.50334667] ,
  [0.44580989, 0.50284511] ,
  [0.5905391,  0.70158824] ,
  [0.40947203, 0.67515642] ,
  [0.65477449, 0.91138141] ,
  [0.36900823, 0.81468474]]
poses =[[[0.46471494,0.23287114], [0.46771983,0.21898333], [0.43287791,0.21873544], [0.49123543,0.23445683], [0.40681923,0.22708205], [0.53842653 ,0.30890295], [0.36190601 ,0.31074432], [0.63287016 ,0.29724553], [0.24799975 ,0.26122134], [0.712948   ,0.22795067], [0.15992125, 0.2112368 ], [0.49331567, 0.52852436], [0.38067907 ,0.52578044], [0.54329794, 0.63695553], [0.3195607,  0.64145361], [0.5511174  ,0.78606388], [0.30908613 ,0.81106502]],
[[ 0.34014858,  0.17379135] ,
  [ 0.36384169,  0.146875  ] ,
  [ 0.31952599,  0.1558199 ] ,
  [ 0.39087816,  0.14501395] ,
  [ 0.29322362,  0.16435769] ,
  [ 0.44942507,  0.18724965] ,
  [ 0.30858383,  0.30288591] ,
  [ 0.56226364,  0.10128063] ,
  [ 0.27425084,  0.44318412] ,
  [ 0.61808442,  0.04765346] ,
  [ 0.18693124,  0.53816655] ,
  [ 0.52387138,  0.4305091 ] ,
  [ 0.44349462,  0.5376528 ] ,
  [ 0.64126189,  0.43481431] ,
  [ 0.46296197,  0.68621739] ,
  [ 0.7743575,   0.54488908] ,
  [0.4,0.8       ]],
  [[0.71604285, 0.15653112] ,
  [0.73024136 ,0.13847823] ,
  [0.70016483 ,0.14083351] ,
  [0.74882728, 0.15032951] ,
  [0.6679834 , 0.14557312] ,
  [0.69454495, 0.1891326 ] ,
  [0.67148455, 0.22281282] ,
  [0.76500489, 0.19755103] ,
  [0.72933125, 0.31764458] ,
  [0.83152196, 0.18370117] ,
  [0.85349953, 0.42007762], 
  [0.5480724 , 0.40672671], 
  [0.49522963, 0.40484684], 
  [0.53180767, 0.6641502 ], 
  [0.36238944, 0.51627727], 
  [0.51392827, 0.83357089], 
  [0.21264142, 0.57210243]],
  [[0.63398223, 0.24198456] ,
  [0.64429612 ,0.24539133] ,
  [0.61825431 ,0.24175507] ,
  [0.67389748 ,0.25693179] ,
  [0.58860262 ,0.25020813] ,
  [0.6863448 , 0.31627936] ,
  [0.56524416, 0.30213088] ,
  [0.75364449, 0.23783373] ,
  [0.53523348, 0.21879462] ,
  [0.68107738, 0.13933097] ,
  [0.62281505, 0.15166613] ,
  [0.62555793, 0.53986093] ,
  [0.53785477, 0.50260955] ,
  [0.59862776, 0.69667728] ,
  [0.45426949, 0.66108209] ,
  [0.57314013, 0.87693882] ,
  [0.40578496, 0.76189476]],
  [[0.33547989, 0.24457643], 
  [0.3607444,  0.22187484] ,
  [0.3285903,  0.22879599] ,
  [0.37916049, 0.24074223] ,
  [0.30665966, 0.24236338] ,
  [0.39769716, 0.34835114] ,
  [0.28100257, 0.34991355] ,
  [0.41512847, 0.48393867] ,
  [0.20662783, 0.38820736] ,
  [0.41944205, 0.60309746] ,
  [0.16268978, 0.38092859] ,
  [0.35332278, 0.59578603] ,
  [0.2856997,  0.59997721] ,
  [0.33935796, 0.78534572] ,
  [0.28880878, 0.78594449] ,
  [0.46116668, 0.87775959] ,
  [0.40908957, 0.9166514 ]],
  [[0.46514611, 0.19599202], 
  [0.48008856, 0.17697187], 
  [0.45583785, 0.17612537], 
  [0.50760405, 0.19988374], 
  [0.43378167, 0.19266993], 
  [0.53080717, 0.30968607], 
  [0.3826808,  0.29333367], 
  [0.61293537, 0.39046255], 
  [0.33993933, 0.43419193], 
  [0.67267425, 0.33253799], 
  [0.29907347, 0.54315715], 
  [0.46091523, 0.57109497], 
  [0.36660117, 0.54770369], 
  [0.43895048, 0.77862662], 
  [0.32117957, 0.74011447], 
  [0.41347956, 0.96183673], 
  [0.32165361, 0.8274765 ]]
  ]
threshold = 10
actual = [[i+random.random()*0.1,j+random.random()*0.1] for i,j in startPose]
print(actual)
def Angle(points,A,B,C):
    if max(A,B,C)<len(points):
        xA,yA = points[A]
        xB,yB = points[B]
        xC,yC = points[C]
        above = (yA-yB)/(xA-xB)*(xC-xB)<yC
        dot = ((xA-xB)*(xC-xB)+(yA-yB)*(yC-yB))/(((xA-xB)**2+(yA-yB)**2)**(1/2))/(((xC-xB)**2+(yC-yB)**2)**(1/2))
        if dot <= -1 or dot >= 1:
            dot = 0
        return (math.acos(dot)*180/math.pi,above)
    return (0,False)

def angle_deviation(reference,actual,A,B,C):  
    angle1, above1 = Angle(reference,A,B,C)
    angle1 = angle1 if above1 else 360-angle1
    angle2, above2 = Angle(actual,A,B,C)
    angle2 = angle2 if above2 else 360-angle2
    return (abs(angle2-angle1))/threshold

Indices = [[5,6,8],[6,8,10],[6,5,7],[5,7,9],[6,12,14],[12,14,16],[5,11,13],[11,13,15]]

start_time = 0
pose_temp = None
def AddPose(statevar,startval,points):
    global pose_temp,poses
    state0 = True
    tempvar = statevar
    if statevar == startval:
        draw_text(img,img_size,"Make A Pose!",1,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,0),"bottom",10)
        for i,j in points:
            if i == -1 or j == -1:
                state0 = False
        if not state0:
            draw_text(img,img_size,"Please show your whole body!",2,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"center",0)
    elif startval+1 <= statevar <  startval+3:
        
        tempvar, texts = timer(tempvar,statevar+1,3)
        draw_text(img,img_size,"Pause! Taking Pose in "+texts+"s",1,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,0),"bottom",10)
    elif startval+3 <= statevar <  startval+4:
        for i,j in points:
            if i == -1 or j == -1:
                state0 = False
        if not state0:
            draw_text(img,img_size,"Please show your whole body!",2,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"center",0)
        else:
            draw_text(img,img_size,"Taken!",2,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"center",0)
            pose_temp = list(points)
            poses.append(pose_temp)
            startvar+=1
    elif startval+4 <= statevar <  startval+6:
        tempvar, texts = timer(tempvar,statevar+1,3)
        draw_text(img,img_size,"Thanks!",1,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,0),"bottom",10)

            

        
def timer(statevar,startval, duration):
    global start_time
    if statevar == startval:
        statevar+=1
        start_time = time.time()
        print(start_time)
    if statevar >= startval + 2:
        pass
    elif time.time()-start_time > duration:
        statevar+=1
    
    #print(time.time(),start_time)
    return statevar

def timer_text(statevar, duration):
    global start_time
    if time.time()-start_time > duration:
        return statevar+1, str(int(duration-time.time()+start_time))
    return statevar, str(int(duration-time.time()+start_time))
    


def Score(reference, actual):
    score = 100
    for i,j,k in Indices:
        score-=angle_deviation(reference,actual,i,j,k)
    return score if score > 0 else 0

def map_bbox_to_image_coords(
   bbox: List[float], image_size: Tuple[int, int]
) -> List[int]:
   """First helper function to convert relative bounding box coordinates to
   absolute image coordinates.
   Bounding box coords ranges from 0 to 1
   where (0, 0) = image top-left, (1, 1) = image bottom-right.

   Args:
      bbox (List[float]): List of 4 floats x1, y1, x2, y2
      image_size (Tuple[int, int]): Width, Height of image

   Returns:
      List[int]: x1, y1, x2, y2 in integer image coords
   """
   width, height = image_size[0], image_size[1]
   x1, y1, x2, y2 = bbox
   x1 *= width
   x2 *= width
   y1 *= height
   y2 *= height
   return int(x1), int(y1), int(x2), int(y2)


def map_keypoint_to_image_coords(
   keypoint: List[float], image_size: Tuple[int, int]
) -> List[int]:
   """Second helper function to convert relative keypoint coordinates to
   absolute image coordinates.
   Keypoint coords ranges from 0 to 1
   where (0, 0) = image top-left, (1, 1) = image bottom-right.

   Args:
      bbox (List[float]): List of 2 floats x, y (relative)
      image_size (Tuple[int, int]): Width, Height of image

   Returns:
      List[int]: x, y in integer image coords
   """
   width, height = image_size[0], image_size[1]
   x, y = keypoint
   x *= width
   y *= height
   return int(x), int(y)


def draw_text(img, x, y, text_str: str, color_code):
   """Helper function to call opencv's drawing function,
   to improve code readability in node's run() method.
   """
   cv2.putText(
      img=img,
      text=text_str,
      org=(x, y),
      fontFace=cv2.FONT_HERSHEY_DUPLEX,
      fontScale=0.4,
      color=color_code,
      thickness=2,
   )



def draw_human_poses(
    image: np.ndarray, keypoints: list, scale
) -> None:
    # pylint: disable=too-many-arguments
    """Draw poses onto an image frame.

    Args:
        image (np.array): image of current frame
        keypoints (List[Any]): list of keypoint coordinates
        keypoints_conns (List[Any]): list of keypoint connections
    """
    image_size = np.array([int(i*scale) for i in list(get_image_size(image))])
    offset = int(50*scale)
    xs = [i for i,j in keypoints]
    ys = [j for i,j in keypoints]
    range_x = max(xs)-min(xs)
    range_y = max(ys)-min(ys)
    keypoints = np.array([[i+0.5-range_x/2-min(xs),j+0.5-range_y/2-min(ys)] for i,j in keypoints])
    cv2.rectangle(image, pt1=(offset,offset), pt2=(image_size[0]-offset,image_size[1]-offset), color=(230,230,230), thickness= -1)
    _draw_connections(image, keypoints, image_size, scale)
    _draw_keypoints(image, keypoints, image_size, TOMATO, POINT_RADIUS, scale)
    num_persons = keypoints.shape[0]



def _draw_connections(
    frame: np.ndarray,
    keypoints: list,
    image_size: Tuple[int, int],scale
) -> None:
    """Draw connections between detected keypoints"""
    ind=[[5,6],[5,7],[7,9],[6,8],[6,10],[6,12],[11,12],[5,11],[12,14],[11,13],[14,16],[13,15]]
    for connection in ind:
        a,b = connection
        pts = np.array([keypoints[a],keypoints[b]])
        pt1, pt2 = project_points_onto_original_image(pts, image_size)
        cv2.line(frame, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0,0,0), int(50*scale))


def _draw_keypoints(
    frame: np.ndarray,
    keypoints: list,
    image_size: Tuple[int, int],
    keypoint_dot_color: Tuple[int, int, int],
    keypoint_dot_radius: int,scale 
) -> None:
    # pylint: disable=too-many-arguments
    """Draw detected keypoints"""
    img_keypoints = project_points_onto_original_image(keypoints, image_size)
    n = 0
    for _, keypoint in enumerate(img_keypoints):
        if n == 0:
            _draw_one_keypoint_dot(frame, keypoint, (255,247,136), int(60*scale))
        #elif n>4:
            #_draw_one_keypoint_dot(frame, keypoint, (0,0,0), keypoint_dot_radius)
        n+=1


def _draw_one_keypoint_dot(
    frame: np.ndarray,
    keypoint: list,
    keypoint_dot_color: Tuple[int, int, int],
    keypoint_dot_radius: int,
) -> None:
    """Draw single keypoint"""
    cv2.circle(
        frame, (keypoint[0], keypoint[1]), keypoint_dot_radius, keypoint_dot_color, -1
    )

def draw_text(img,img_size,text,thickness,font,fontscale, color,position,offset):
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    # get coords based on boundary
    if position == "center":
        textX = (img_size[0] - textsize[0]) / 2
        textY = (img_size[1] + textsize[1]) / 2
    elif position == "bottom":
        textX = (img_size[0] - textsize[0]) / 2
        textY = img_size[1]- offset
    elif position == "top":
        textX = (img_size[0] - textsize[0]) / 2
        textY = (textsize[1]) + offset
    elif position == "top-right":
        textX = (img_size[0] - textsize[0])-offset
        textY = textsize[1]+offset
    elif position == "bottom-left":
        textX = offset
        textY = img_size[1]- offset -100
    cv2.rectangle(img, pt1=(int(textX-20),int(textY+20)), pt2=(int(textX+textsize[0]+20),int(textY-textsize[1]-20)), color=(0,0,0), thickness= -1)
    cv2.putText(img, text, (int(textX), int(textY) ), font, 1, color, thickness)

class Node(AbstractNode):
   global tens, start, poses
   """Custom node to display keypoints and count number of hand waves

   Args:
      config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
   """

   def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
      super().__init__(config, node_path=__name__, **kwargs)
      # setup object working variables
      self.state =0
      self.moves = None
      self.success = False

   def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
      global tens, start
      """This node draws keypoints and count hand waves.

      Args:
            inputs (dict): Dictionary with keys
               "img", "bboxes", "bbox_scores", "keypoints", "keypoint_scores".

      Returns:
            outputs (dict): Empty dictionary.
      """

      # get required inputs from pipeline
      img = inputs["img"]
      bboxes = inputs["bboxes"]
      bbox_scores = inputs["bbox_scores"]
      keypoints = inputs["keypoints"]
      keypoint_scores = inputs["keypoint_scores"]
      print(keypoints[0])
      img_size = (img.shape[1], img.shape[0])  # image width, height
      if time.time() - tens*10>10:
        tens+=1
        #print(keypoints)
      # get bounding box confidence score and draw it at the
      # left-bottom (x1, y2) corner of the bounding box (offset by 30 pixels)
      state0=True
      if self.state == 0:
        draw_human_poses(inputs["img"], np.array(startPose),0.3)
        draw_text(img,img_size,"Copy the pose and reach 90 to Start",2,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"center",0)
      elif 1 <= self.state <3:
        self.state = timer(self.state,1,2)
        draw_text(img,img_size,"Great Job!",2,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"center",0)
      elif 3 <= self.state <5:
        self.state = timer(self.state,3,2)
        draw_text(img,img_size,"Round 1",2,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"center",0)
        self.success = False
      elif 5 <= self.state <6:
        self.moves = random.sample(poses,3)
        self.state +=1
      elif 6 <= self.state <8:
        self.state = timer(self.state,6,1)
        draw_human_poses(inputs["img"], np.array(self.moves[0]),1)
      elif 8 <= self.state <9:
        draw_human_poses(inputs["img"], np.array(self.moves[0]),0.3)
        self.state, texts = timer_text(self.state,10)
        draw_text(img,img_size,texts,1,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"bottom-left",20)
      elif (9 <= self.state <11) and self.success:
        self.state = timer(self.state,9,2)
        draw_text(img,img_size,"Great Job!",2,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"center",0)
      elif (9 <= self.state <11) and not(self.success):
        self.state = timer(self.state,9,2)
        draw_text(img,img_size,"Time's Out!",2,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"center",0)
      elif 11 <= self.state <13:
        self.state = timer(self.state,11,2)
        draw_text(img,img_size,"Round 2",2,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"center",0)
        self.success = False
      elif 13 <= self.state <15:
        self.state = timer(self.state,13,1)
        draw_human_poses(inputs["img"], np.array(self.moves[1]),1)
      elif 15 <= self.state <16:
        draw_human_poses(inputs["img"], np.array(self.moves[1]),0.3)
        self.state, texts = timer_text(self.state,10)
        draw_text(img,img_size,texts,1,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"bottom-left",20)
      elif (16 <= self.state <18) and self.success:
        self.state = timer(self.state,16,2)
        draw_text(img,img_size,"Great Job!",2,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"center",0)
      elif (16 <= self.state <18) and not self.success:
        self.state = timer(self.state,16,2)
        draw_text(img,img_size,"Time's Out!",2,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"center",0)
      elif 18 <= self.state <20:
        self.state = timer(self.state,18,2)
        draw_text(img,img_size,"Round 3",2,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"center",0)
      elif 20 <= self.state <22:
        self.state = timer(self.state,20,1)
        draw_human_poses(inputs["img"], np.array(self.moves[2]),1)
      elif 22 <= self.state <23:
        draw_human_poses(inputs["img"], np.array(self.moves[2]),0.3)
        self.state, texts = timer_text(self.state,10)
        draw_text(img,img_size,texts,1,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"bottom-left",20)
      elif 23 <= self.state <25:
        self.state = timer(self.state,23,2)
        draw_text(img,img_size,"Finished",2,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"center",0)
      elif 25 <= self.state <27:
        self.state = timer(self.state,25,3)
        draw_text(img,img_size,"Congratulations!",2,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"center",0)
      elif 27 <= self.state <29:
        self.state = timer(self.state,27,1)
        draw_text(img,img_size,"Make a pose!",2,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"center",0)
      elif 29 <= self.state <30:
        draw_text(img,img_size,"Make A Pose!",1,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,0),"bottom",10)
        for i,j in list(inputs["keypoints"][0]):
            if i == -1 or j == -1:
                state0 = False
        if not state0:
            draw_text(img,img_size,"Please show your whole body!",2,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"center",0)
        else:
            self.state+=1
      elif 30 <= self.state <  31:
        for i,j in list(inputs["keypoints"][0]):
            if i == -1 or j == -1:
                state0 = False
        if not state0:
            draw_text(img,img_size,"Please show your whole body!",2,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"center",0)
        self.state, texts = timer_text(self.state,5)
        draw_text(img,img_size,"Pause! Taking Pose in "+texts+"s",1,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,0),"bottom",10)
      elif 31 <= self.state <  32:
        for i,j in list(inputs["keypoints"][0]):
            if i == -1 or j == -1:
                state0 = False
        if not state0:
            draw_text(img,img_size,"Please show your whole body!",2,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"center",0)
        else:
            draw_text(img,img_size,"Taken!",2,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"center",0)
            pose_temp = list(inputs["keypoints"][0])
            poses.append(pose_temp)
            self.state+=1
      elif 32 <= self.state <  34:
        self.state = timer(self.state,32,1)
        draw_text(img,img_size,"Taken!",2,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"center",0)
      elif 34 <= self.state <  36:
        self.state = timer(self.state,34,5)
        draw_text(img,img_size,"Thanks!",1,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,0),"bottom",10)
      elif self.state >=36:
        self.state = 0
      
      print(self.state)

        
      
      if len(bboxes) < 1 or len(bbox_scores) < 1:
        draw_text(img,img_size,"No Person Detected.",1,cv2.FONT_HERSHEY_DUPLEX,1, (0,0,200),"bottom",30)
      else:

        score = round(Score(startPose,keypoints[0]),0)     
        draw_text(img,img_size,str(score),1,cv2.FONT_HERSHEY_DUPLEX,1, (0,255,255),"top-right",30)
        if (self.state == 0 or self.state == 8 or self.state == 15 or self.state == 22) and score > 90:
            self.state += 1
            self.success = True


      return {}

