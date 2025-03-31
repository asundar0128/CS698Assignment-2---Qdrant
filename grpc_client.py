import grpc
import yolo_search_pb2
import yolo_search_pb2_grpc

# Create gRPC channel and stub
generatedChannelValue = grpc.insecure_channel('localhost:50051')
generatedStub = yolo_search_pb2_grpc.YoloSearchStub(generatedChannelValue)

# Prepare request
generatedRequestValue = yolo_search_pb2.SearchRequest(vector=[100, 200, 300, 400], top_k=5)

# Make the RPC call
generatedResponseValue = generatedStub.generatedObjectSearch(generatedRequestValue)

# Process and print results
for generatedObjectValue in generatedResponseValue.objects:
    print(f"The following object has been found: {generatedObjectValue.label} "
          f"with the following confidence {generatedObjectValue.confidence} "
          f"at {generatedObjectValue.coordinates}")
