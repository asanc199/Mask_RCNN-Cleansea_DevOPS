import cv2
from visualize_cv2 import model, display_instances, class_names
import sys

args = sys.argv

if(len(args) < 2):
    print("\n ERROR -> Run: python video_demo.py 0 or video_file_path\n")
    sys.exit(0)
name=args[1]

if(len(args[1]) == 1):
    name = int(args[1])

name = r"D:\Cleansea\cleansea_dataset\Videos\video_analisis\debrisVideo2.mp4"
stream = cv2.VideoCapture(name)

while True:
    ret, frame = stream.read()
    if not ret:
        print("unable to fetch frame")
        break
    results = model.detect([frame], verbose = 1)

    #Visualize Results
    r = results[0]

    masked_image =  display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    cv2.imshow("Deteccion_Residuos",masked_image)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

stream.release()
cv2.destroyAllWindows("Deteccion_Residuos")