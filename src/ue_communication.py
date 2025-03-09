import socket
import threading
import json
import queue
import time
from event_manager import EventManager

class UECommunicationBase:
    def __init__(self, host='localhost', port=9998):
        self.host = host
        self.port = port
        self.command_queue = queue.Queue()
        self.event_manager = EventManager()
        self.running = False
        
    def get_commands(self):
        """获取从UE发送的命令"""
        commands = []
        while not self.command_queue.empty():
            commands.append(self.command_queue.get())
        return commands
        
    def parse_message(self, message):
        """解析从UE接收到的消息"""
        try:
            data = json.loads(message)
            if data['type'] == 'command':
                # 将命令添加到队列
                self.command_queue.put(data['data'])
            elif data['type'] == 'event':
                # 触发事件
                self.event_manager.post_event(data['event_name'], data['event_data'])
        except Exception as e:
            print(f"Error parsing message: {e}")


class UECommunicationServer(UECommunicationBase):
    def __init__(self, host='localhost', port=9998):
        super().__init__(host, port)
        self.clients = []
        self.clients_lock = threading.Lock()
        
    def start(self):
        """启动UE通信服务器"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True
        
        # 启动接受客户端连接的线程
        threading.Thread(target=self.accept_clients, daemon=True).start()
        print(f"UE Communication Server started on {self.host}:{self.port}")
        
    def accept_clients(self):
        """接受UE客户端连接"""
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                print(f"UE client connected: {addr}")
                
                with self.clients_lock:
                    self.clients.append(client_socket)
                
                # 为每个客户端启动一个接收线程
                threading.Thread(target=self.handle_client, args=(client_socket, addr), daemon=True).start()
            except Exception as e:
                print(f"Error accepting client: {e}")
                if not self.running:
                    break
                time.sleep(1)
    
    def handle_client(self, client_socket, addr):
        """处理UE客户端连接"""
        buffer = ""
        while self.running:
            try:
                data = client_socket.recv(4096).decode('utf-8')
                if not data:
                    break
                    
                buffer += data
                
                # 处理可能的多条消息
                while '\n' in buffer:
                    message, buffer = buffer.split('\n', 1)
                    self.parse_message(message)
                    
            except Exception as e:
                print(f"Error handling client {addr}: {e}")
                break
                
        print(f"UE client disconnected: {addr}")
        with self.clients_lock:
            if client_socket in self.clients:
                self.clients.remove(client_socket)
        client_socket.close()
    
    def send_message(self, message):
        """向所有UE客户端发送消息"""
        if not message.endswith('\n'):
            message += '\n'
            
        with self.clients_lock:
            disconnected_clients = []
            for client in self.clients:
                try:
                    client.sendall(message.encode('utf-8'))
                except Exception as e:
                    print(f"Error sending message to client: {e}")
                    disconnected_clients.append(client)
            
            # 移除断开连接的客户端
            for client in disconnected_clients:
                if client in self.clients:
                    self.clients.remove(client)
                    client.close()
    
    def stop(self):
        """停止UE通信服务器"""
        self.running = False
        with self.clients_lock:
            for client in self.clients:
                try:
                    client.close()
                except:
                    pass
            self.clients.clear()
        
        try:
            self.server_socket.close()
        except:
            pass


class UECommunicationClient(UECommunicationBase):
    def __init__(self, host='localhost', port=9998):
        super().__init__(host, port)
        self.socket = None
        self.reconnect_interval = 5  # 重连间隔（秒）
        
    def start(self):
        """启动UE通信客户端"""
        self.running = True
        self.connect()
        
    def connect(self):
        """连接到UE通信服务器"""
        while self.running:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
                print(f"Connected to UE server at {self.host}:{self.port}")
                
                # 启动接收线程
                self.receive_thread = threading.Thread(target=self.receive_messages, daemon=True)
                self.receive_thread.start()
                
                # 等待接收线程结束（断开连接）
                self.receive_thread.join()
                
                # 如果程序还在运行，尝试重新连接
                if self.running:
                    print(f"Connection lost. Reconnecting in {self.reconnect_interval} seconds...")
                    time.sleep(self.reconnect_interval)
            except Exception as e:
                print(f"Failed to connect to UE server: {e}")
                if self.running:
                    print(f"Retrying in {self.reconnect_interval} seconds...")
                    time.sleep(self.reconnect_interval)
    
    def receive_messages(self):
        """接收来自UE服务器的消息"""
        buffer = ""
        while self.running:
            try:
                data = self.socket.recv(4096).decode('utf-8')
                if not data:
                    break
                    
                buffer += data
                
                # 处理可能的多条消息
                while '\n' in buffer:
                    message, buffer = buffer.split('\n', 1)
                    self.parse_message(message)
                    
            except Exception as e:
                print(f"Error receiving message: {e}")
                break
        
        # 关闭套接字
        try:
            self.socket.close()
        except:
            pass
    
    def send_message(self, message):
        """向UE服务器发送消息"""
        if not message.endswith('\n'):
            message += '\n'
            
        try:
            if self.socket:
                self.socket.sendall(message.encode('utf-8'))
        except Exception as e:
            print(f"Error sending message to UE server: {e}")
            # 连接断开，关闭套接字
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
    
    def stop(self):
        """停止UE通信客户端"""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass