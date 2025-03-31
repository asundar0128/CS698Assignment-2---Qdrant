import grpc
from concurrent import futures
import yolo_search_pb2
import yolo_search_pb2_grpc
from qdrant_client import QdrantClient

# Initialize Qdrant Client
generatedClientValue = QdrantClient(host="localhost", port=6333)

class generatedSearchService(yolo_search_pb2_grpc.YoloSearchServicer):
    def generatedObjectSearch(self, generatedRequestValue, generatedContextValue):
        generatedResultsValue = generatedClientValue.search(
            collection_name="objects",
            query_vector=list(generatedRequestValue.vector),
            limit=generatedRequestValue.top_k
        )

        generatedResponse = yolo_search_pb2.SearchResponse()
        for generatedResult in generatedResultsValue:
            generatedObject = generatedResponse.objects.add()
            generatedObject.id = generatedResult.id
            generatedObject.coordinates.extend(generatedResult.vector)
            generatedObject.confidence = float(generatedResult.payload.get("confidence", 0.0))
            generatedObject.label = generatedResult.payload.get("label", "")
        
        return generatedResponse

def generatedService():
    generatedServerValue = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    yolo_search_pb2_grpc.add_YoloSearchServicer_to_server(generatedSearchService(), generatedServerValue)
    generatedServerValue.add_insecure_port('[::]:50051')
    generatedServerValue.start()
    print("gRPC YOLO Search Service is running on port 50051...")
    generatedServerValue.wait_for_termination()

if __name__ == '__main__':
    generatedService()
