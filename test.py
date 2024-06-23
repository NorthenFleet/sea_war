class A():
    def __init__(self):
        self.range = 10
        self.capacity = 100


class entity():
    def __init__(self, A):
        self.weapon = A()

    def attack(self):
        self.weapon.capacity -= 10


if __name__ == '__main__':
    a = A()
    b = entity(a)
    c = entity(a)

    b.attack()
    b.attack()
    print(b.weapon.capacity)
    print(c.weapon.capacity)
