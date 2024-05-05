class GameServiceServicer(gamelogic_pb2_grpc.GameServiceServicer):
    def GameChannel(self, request_iterator, context):
        for new_action in request_iterator:
            # ���ݶ���������Ϸ״̬
            # ...
            yield gamelogic_pb2.GameResponse(status='״̬����')
