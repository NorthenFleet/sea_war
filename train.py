import grpc
from grpcpackage import game_pb2 as pb2
from grpcpackage import game_pb2_grpc as pb2_grpc


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


class Train():
    def __init__(self) -> None:
        pass

    def run(self):
        pass


if __name__ == '__main__':
    Train().run()
