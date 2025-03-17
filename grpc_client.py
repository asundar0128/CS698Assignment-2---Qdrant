import grpc
import yolo_search_pb2
import yolo_search_pb2_grpc

generatedChannelValue = grpc.insecure_channel('localhost:50051')
generatedStub = yolo_search_pb2_grpc.YoloSearchStub(generatedChannelValue)

generatedRequestValue = yolo_search_pb2.SearchRequest(vector=[100, 200, 300, 400], top_k=5)
generatedResponseValue = stub.SearchObjects(generatedRequestValue)

for generatedObjectValue in generatedResponseValue.objects:
    print(f"The following object has been found: {generatedObjectValue.label} with the following confidence {generatedObjectValue.confidence} at {generatedObjectValue.coordinates}")
