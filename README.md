A complete pipeline with ROS2, OpenCV, YOLOv5, and Qdrant is set up in this project from beginning to finish. Camera streaming, object detection, vector
storage, and semantic querying with gRPC are all used.

In fact, Remote Procedure Calls were used in entirety to not have to load everything into Docker on a single container. 

Loading everything into Docker in one single container made the outputs never appear and take excessive amounts of time. 

camera_publisher_node.py

Publishes GStreamer camera feed to /camera/image_raw. 

gstreamer_camera.py

Publishes GStreamer camera feed for a GStreamer-specific node.

yolo_detector_node.py

Helps with subscribing the ROS2 Topic and detecting camera with log files.

detect.py

Iterates through YOLOv5, draws the bounding boxes, and stores the vector results in a Qdrant Database. 

maze_detection.py

Visualizes robots in a maze while detecting objects and updating the Qdrant database. 

grpc_server.py

Exposing detected vectors in Qdrant database via gRPC using Remote Procedure Calls, where multiple APIs are invoked in the Docker container
instead of a single container overwhelmed with excessive vector data points from the Qdrant database.

grpc_client.py

gRPC server is queried to retrieve stored objects based on how close a respective object is in the Qdrant database: umbrella, cup, book, etc.
Closer objects, or those with more proximity will be queried with gRPC server and returned to the client visually. 

test_camera.py

Testing to ensure that openCV is capturing directly from the webcam
