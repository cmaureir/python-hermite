from plummer import Plummer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

h = 0.05
number = 10
# p = Plummer(N=number, step=h, PlummerRadius=1, seed=10)
p = Plummer(N=number, step=h, PlummerRadius=1)
print("model initialized")
El = []
Tl = []
t = 0
fig = plt.figure();
ax = fig.add_subplot(projection="3d")
ax.set(xlabel="X")
ax.set(ylabel="Y")
ax.set(zlabel="z")
l = []

def animate():
    global t
    # print(t)
    #plt.figure(clear="True")
    #plt.plot(p.Position, label="Lz")
    #plt.show()
    #Ll.append(p.l())
    p.stepHermite4()
    t +=h
    ax.clear()
    temp = p.Position.T
    # ax.plot(temp[0],temp[1],'o', label="time: {:.5f}, energy: {:0.5f} ".format(t,p.energy()))
    # ax.scatter3D(temp[0],temp[1], temp[2],'o', label="time: {:.5f}, energy: {:0.5f} ".format(t,p.energy()))
    ax.scatter(temp[0],temp[1], temp[2],color="black", label="time: {:.5f}, energy: {:0.5f} ".format(t,p.energy()))
    ax.legend()
    return temp
i = t
a = animate()
def lines(i):
    global l
    a = animate()
    l.append(a)
    if(len(l)>1000):
        l.pop(0)

    temp = np.array(l)
    # ax.plot(a[0], a[1],a[2])
    temp = temp.T
    for j in range(len(temp)):
        ax.plot(temp[j][0], temp[j][1], temp[j][2])


ani = FuncAnimation(fig,  lines,  cache_frame_data=False, interval=1, repeat=False)
# ani = FuncAnimation(fig, animate,cache_frame_data=False, repeat=False)
plt.show()
