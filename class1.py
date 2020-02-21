class First:
    def setdata(self,value):
        self.data = value
    def display(self):
        print(self.data)

class Second(First):
    def display(self):
        print('current value = "%s"' % self.data)

#class Third(Second):
#    def display(self):
#        print('overload value = "%s"' % self.data)
#z2 = Third()
#z2.setdata(55)
#z2.display()

class Third(Second):
    def __init__(self, value):
        self.data = value
    def __add__(self, other):
        return Third(self.data + other)
    def __str__(self):
        return '[Third: %s]' %self.data
    def mul(self, other):
        self.data *= other

a = Third('abc')        
a.display()

z = Second()
z.setdata(54)
z.display()
