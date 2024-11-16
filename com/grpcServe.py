from concurrent import futures
import grpc
from grpcpackage import game_pb2
from grpcpackage import game_pb2_grpc

class GameService(game_pb2_grpc.GameServiceServicer):
    def ControlCommand(self, request, context):
        # 根据request.command处理游戏逻辑
        # 这里简单模拟返回游戏状态
        response = game_pb2.GameState(score=123, game_over=False)
        return response

    def GameStateStream(self, request, context):
        # 根据请求决定是否发送游戏状态流
        if request.subscribe:
            while True:
                yield game_pb2.GameState(score=123, game_over=False, additional_info="Streaming...")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    game_pb2_grpc.add_GameServiceServicer_to_server(GameService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
