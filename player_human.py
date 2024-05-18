from pynput import keyboard, mouse

class HumanPlayer(Base_player):
    def __init__(self, name):
        super().__init__(name)
        self.current_action = None
        self.default_action = 'idle'
        self.start_listening()

    def start_listening(self):
        def on_press(key):
            try:
                if key.char == 'w':
                    self.current_action = 'move_up'
                elif key.char == 's':
                    self.current_action = 'move_down'
                elif key.char == 'a':
                    self.current_action = 'move_left'
                elif key.char == 'd':
                    self.current_action = 'move_right'
                elif key.char == ' ':
                    self.current_action = 'attack'
            except AttributeError:
                pass

        def on_click(x, y, button, pressed):
            if pressed:
                self.current_action = 'attack'

        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()

        self.mouse_listener = mouse.Listener(on_click=on_click)
        self.mouse_listener.start()

    def choose_action(self, state):
        action = self.current_action if self.current_action is not None else self.default_action
        self.current_action = None
        return action

    def stop_listening(self):
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        if self.mouse_listener:
            self.mouse_listener.stop()

    def receive_state_update(self, state):
        for agent in self.agents:
            agent.choose_action(state)
