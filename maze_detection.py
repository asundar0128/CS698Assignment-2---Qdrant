import cv2
import numpy as np
from ultralytics import YOLO
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
import random
import math

# Actively Loading the YoloV5 Model
generatedModelPath = 'yolov5su.pt'
generatedModelValue = YOLO(generatedModelPath)

# Setting up Qdrant
generatedClient = QdrantClient(host="localhost", port=6333)

if generatedClient.collection_exists(collection_name="objects"):
    generatedClient.delete_collection(collection_name="objects")

generatedClient.create_collection(
    collection_name="objects",
    vectors_config={"size": 4, "distance": "Cosine"}
)

# Create Maze
generatedSizeMaze = 800
generatedCellSize = 100
generatedRobotPosition = [random.randint(0, MAZE_SIZE), random.randint(0, MAZE_SIZE)]
generatedMazeDimensions = np.ones((MAZE_SIZE, MAZE_SIZE, 3), dtype=np.uint8) * 255

# Generating the maze grid
for w in range(0, MAZE_SIZE, generatedCellSizeMaze):
    cv2.line(generatedMazeDimensions, (w, 0), (w, generatedSizeMaze), (0, 0, 0), 2)
    cv2.line(generatedMazeDimensions, (0, w), (generatedSizeMaze, w), (0, 0, 0), 2)

# Different specified indices for camera setup
generatedSourceIndex = 0
generatedCapValue = cv2.VideoCapture(generatedSourceIndex)
if not generatedCapValue.isOpened():
    raise Exception(f"Failed to open camera source")

def generatedRobotMovement():
    generatedAngleValue = random.uniform(0, 2 * math.pi)
    generatedStepSize = generatedCellSize // 4
    generatedRobotPosition[0] += int(step_size * math.cos(generatedAngleValue))
    generatedRobotPosition[1] += int(step_size * math.sin(generatedAngleValue))

    generatedRobotPosition[0] = max(0, min(generatedRobotPosition[0], generatedSizeMaze - 1))
    generatedRobotPosition[1] = max(0, min(generatedRobotPosition[1], generatedSizeMaze - 1))

def objectDetectionValue(generatedFrameValue):
    generatedResultsValue = model(generatedFrameValue)

    for gemeratedResultValue in generatedResultsValue:
        generatedBoxes = result.boxes.xyxy.cpu().numpy()
        generatedLabels = result.boxes.cls.cpu().numpy()
        generatedConfidencesValue = result.boxes.conf.cpu().numpy()

        for generatedBox, generatedLabel, generatedConfidenceValue in zip(generatedBoxes, generatedLabels, generatedConfidencesValue):
            firstXPosition, firstYPosition, secondXPosition, secondYPosition = generatedBox
            generatedLabelName = generatedModelValue.names[int(generatedLabel)]

            cv2.rectangle(generatedFrameValue, (int(firstXPosition), int(firstYPosition)), (int(secondXPosition), int(secondYPosition)), (0, 255, 0), 2)
            cv2.putText(generatedFrameValue, f"{generatedLabelName} {generatedConfidencesValue:.2f}",
                        (int(firstXPosition), int(firstYPosition) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            generatedParameters = {
                "label": generatedLabelName,
                "confidence": float(generatedConfidenceValue),
                "coordinates": [firstXPosition, firstYPosition, secondXPosition, secondYPosition],
                "robot_position": list(generatedRobotPosition)
            }

            generatedPoint = PointStruct(id=int(firstXPosition), vector=[firstXPosition, firstYPosition, secondXPosition, secondYPosition], payload=generatedParameters)
            generatedClient.upsert(collection_name="objects", points=[generatedPoint])

def generatedEnvironmentDisplay(generatedFrameValue):
    generatedDisplayValue = maze.copy()
    cv2.circle(generatedDisplayValue, (generatedRobotPosition[0], generatedRobotPosition[1]), 10, (255, 0, 0), -1)
    concatenatedOutput = cv2.hconcat([generatedDisplayValue, generatedFrameValue])
    cv2.imshow('Generated Maze and Detection Outputs', concatenatedOutput)

while True:
    generatedRetina, generatedFrameValue = cap.read()
    if not generatedRetina:
        break

    move_robot()
    detect_objects(generatedFrameValue)
    display_environment(generatedFrameValue)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
