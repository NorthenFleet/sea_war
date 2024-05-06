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



def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = pb2_grpc.GameServiceStub(channel)


def action(self):
    user_input = input("Enter a command: ")
    if user_input == "START":
        command = pb2.GameCommand(command=pb2.GameCommand.START)
    elif user_input == "PAUSE":
        command = pb2.GameCommand(command=pb2.GameCommand.PAUSE)
    elif user_input == "RESTART":
        state = pb2.GameCommand(command=pb2.GameCommand.RESTART)
    elif user_input == "END":
        command = pb2.GameCommand(command=pb2.GameCommand.END)
    elif user_input == "MOVE_UP":
        command = pb2.GameCommand(command=pb2.GameCommand.MOVE_UP)
    elif user_input == "MOVE_DOWN":
        command = pb2.GameCommand(command=pb2.GameCommand.MOVE_DOWN)
    elif user_input == "MOVE_LEFT":
        command = pb2.GameCommand(command=pb2.GameCommand.MOVE_LEFT)
    elif user_input == "MOVE_RIGHT":
        command = pb2.GameCommand(command=pb2.GameCommand.MOVE_RIGHT)
    elif user_input == "FIRE":
        command = pb2.GameCommand(command=pb2.GameCommand.FIRE)
    else:
        print("Invalid command. Please try again.")