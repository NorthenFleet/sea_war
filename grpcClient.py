import grpc
from grpcpackage import game_pb2
from grpcpackage import game_pb2_grpc


def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = game_pb2_grpc.GameServiceStub(channel)
        response = stub.SendCommand(game_pb2.GameCommand(command=game_pb2.GameCommand.START))
        print("Game state received: ", response.score, response.game_over)

def generate_actions():
    actions = ['up', 'down', 'left', 'right', 'fire']
    for action in actions:
        yield game_pb2.ActionRequest(action=action)

if __name__ == '__main__':
    run()