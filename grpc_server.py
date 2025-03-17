import grpc
from concurrent import futures
import yolo_search_pb2
import yolo_search_pb2_grpc
from qdrant_client import QdrantClient

generatedClientValue = QdrantClient(host="localhost", port=6333)

class generatedSearchService(yolo_search_pb2_grpc.YoloSearchServicer):
    def generatedObjectSearch(self, generatedRequestValue, generatedContextValue):
        generatedResultsValue = generatedClientValue.search(
            collection_name="objects",
            query_vector=list(request.vector),
            limit=request.top_k
        )
        generatedResponse = yolo_search_pb2.SearchResponse()
        for generatedResult in generatedResultsValue:
            generatedObject = response.objects.add()
            generatedObject.id = result.id
            generatedObject.coordinates.extend(result.vector)
            generatedObject.confidence = result.payload["confidence"]
            generatedObject.label = result.payload["label"]
        return generatedResponse

def generatedService():
    generatedServerValue = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    yolo_search_pb2_grpc.add_YoloSearchServicer_to_server(SearchService(), generatedServerValue)
    generatedServerValue.add_insecure_port('[::]:50051')
    generatedServerValue.start()
    generatedServerValue.wait_for_termination()

if __name__ == '__main__':
    generatedService()
