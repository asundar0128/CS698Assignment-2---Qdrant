import argparse
import cv2
import torch
from ultralytics import YOLO
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

# Initialize Qdrant Client
generatedClientValue = QdrantClient(host="localhost", port=6333)

generatedClientValue.recreate_collection(
    collection_name="objects",
    vectors_config={"size": 4, "distance": "Cosine"}
)

# Load YOLOv5 model
generatedModelPath = 'yolov5s.pt'
generatedModelValue = YOLO(generatedModelPath)

def detect(generatedSourceInput):
    print(f"Opening camera source: {generatedSourceInput}")

    # Open camera stream using GStreamer or OpenCV
    generatedCap = cv2.VideoCapture(
        generatedSourceInput, cv2.CAP_GSTREAMER if isinstance(generatedSourceInput, str) and 'v4l2src' in generatedSourceInput else 0
    )
    
    if not generatedCap.isOpened():
        raise Exception(f"The following device failed to open source at: {generatedSourceInput}")

    while True:
        generatedRetina, generatedFrameValue = generatedCap.read()
        if not generatedRetina:
            print("The device could not grab the following frame.")
            break
        
        # Run YOLOv5 on frame
        generatedResultsOutput = generatedModelValue(generatedFrameValue)

        for generatedResultOutput in generatedResultsOutput:
            generatedBoxes = generatedResultOutput.boxes.xyxy.cpu().numpy()
            generatedLabels = generatedResultOutput.boxes.cls.cpu().numpy()
            generatedConfidences = generatedResultOutput.boxes.conf.cpu().numpy()

            for generatedBox, generatedLabel, generatedConfidence in zip(generatedBoxes, generatedLabels, generatedConfidences):
                generatedXFirstPosition, generatedYFirstPosition, generatedXSecondPosition, generatedYSecondPosition = generatedBox

                # Draw bounding box and label
                cv2.rectangle(generatedFrameValue, 
                              (int(generatedXFirstPosition), int(generatedYFirstPosition)), 
                              (int(generatedXSecondPosition), int(generatedYSecondPosition)), 
                              (0, 255, 0), 2)
                
                cv2.putText(generatedFrameValue, 
                            f"{generatedModelValue.names[int(generatedLabel)]} {generatedConfidence:.2f}",
                            (int(generatedXFirstPosition), int(generatedYFirstPosition) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 255, 0), 2)

                # Store object in Qdrant
                generatedParameters = {
                    "label": generatedModelValue.names[int(generatedLabel)],
                    "confidence": float(generatedConfidence),
                    "coordinates": [generatedXFirstPosition, generatedYFirstPosition, generatedXSecondPosition, generatedYSecondPosition]
                }

                generatedPointDataset = PointStruct(
                    id=int(generatedXFirstPosition),  # use a reproducible int-based ID
                    vector=[float(generatedXFirstPosition), float(generatedYFirstPosition),
                            float(generatedXSecondPosition), float(generatedYSecondPosition)],
                    payload=generatedParameters
                )

                generatedClientValue.upsert(collection_name="objects", points=[generatedPointDataset])

                print(f"Detected {generatedModelValue.names[int(generatedLabel)]} with generatedConfidence {generatedConfidence:.2f} at [{generatedXFirstPosition}, {generatedYFirstPosition}, {generatedXSecondPosition}, {generatedYSecondPosition}]")

        # Show annotated frame
        cv2.imshow('YOLOv5 Detection', generatedFrameValue)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    generatedCap.release()
    cv2.destroyAllWindows()

def main():
    generatedParser = argparse.ArgumentParser()
    generatedParser.add_argument('--source', type=str, default='0', help='Camera source')
    generatedArguments = generatedParser.parse_args()
    
    generatedSourceInput = generatedArguments.source
    if generatedSourceInput.isdigit():
        generatedSourceInput = int(generatedSourceInput)

    detect(generatedSourceInput)

if __name__ == '__main__':
    main()
