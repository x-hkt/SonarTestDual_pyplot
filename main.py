from collections import namedtuple

Pt = namedtuple("Pt", ["lossQ", "theta"])

def proc_xdata_line_unit(xdata_line):
    [x_b, a1, y_b, a2] = xdata_line.split()
    assert a1 == "@Angle"
    assert a2 == "<Entry>"
    x = None
    if x_b[1:-1] == "b#True":
        x = True
    if x_b[1:-1] == "b#False":
        x = False
    assert x != None
    y = float(y_b[1:-1])
    return Pt(x, y)

def proc_xdata_read(fname):
    readf = None
    with open(fname) as f:
        readf = list(filter(lambda x: x != "", map(lambda x: x.strip(), f.readlines())))
    return list(map(
        proc_xdata_line_unit, readf
    ))

r_SIMPLE = proc_xdata_read("results_SIMPLE.txt")
r_COMPLEX = proc_xdata_read("results_COMPLEX.txt")

print("| r_SIMPLE\tlength= %s" % len(r_SIMPLE))
print("| r_COMPLEX\tlength= %s" % len(r_COMPLEX))

import operator as op

def splat(fn):
    return lambda x: fn(*x)

def div_safe_none(a, b):
    if b == 0:
        return None
    else:
        return a / b

def points_derive_losscount(ptsOriginal, minmax= (57, 87), binsize= 3):
    assert (minmax[0] % binsize) == 0
    assert (minmax[1] % binsize) == 0
    pts = filter(lambda x: (x.theta > minmax[0]) and (x.theta < minmax[1]), ptsOriginal)
    #- DBG w"python iterators suck" print("@ actual length = %s" % len(list(pts)))
    bins_value = list(map(lambda _: 0, range((minmax[1] - minmax[0]) // binsize)))
    bins_total = bins_value[:]
    for i in pts:
        curr_index_binref = int((i.theta - minmax[0]) // binsize)
        bins_total[curr_index_binref] += 1
        if i.lossQ:
            bins_value[curr_index_binref] += 1
    return list(map(
        splat(div_safe_none), zip(bins_value, bins_total)
    ))

print("> get: data1 <- r_SIMPLE ...")
data1 = points_derive_losscount(r_SIMPLE)   # data1 <- r_SIMPLE
print("> get: data1 <- r_COMPLEX ...")
data2 = points_derive_losscount(r_COMPLEX)  # data2 <- r_COMPLEX
print("> all done.")

import numpy as np
import matplotlib.pyplot as plt

f = 3
adjq = 57

#- FOR ALTERNATIVE width = 0.3*f

plt.figure(dpi= 150)

#- DBG print((data1, data2))

mkpercent = lambda xs: list(map(lambda sub: sub * 100, xs))

plt.plot(((np.arange(len(data1)) * f) + adjq), mkpercent(data1)) #- FOR ALTERNATIVE width=width
plt.plot(((np.arange(len(data2)) * f) + adjq), mkpercent(data2)) #- FOR ALTERNATIVE width=width

fontsize = 12
plt.title("SONAR Loss vs Surface Angle", fontsize= fontsize+4)
plt.xlabel("Surface Angle (degrees)", fontsize=fontsize)
plt.ylabel("% Loss", fontsize=fontsize)
plt.xticks(range(57, 87, 3), fontsize=fontsize)
plt.yticks(range(0, 70, 10), fontsize=fontsize)
plt.legend(["3.5cm (Simple)", "2.4cm (Complex)"], loc= 4, fontsize=fontsize)

# the angle with matching normals
plt.axvline(66, 0, 60, c= "black")
d = 5.8
plt.text(66-d, 60, "matching", fontsize=fontsize)
plt.text(66-d, 60-5, "normals →", fontsize=fontsize)

plt.savefig("save2D.png")
#> plt.show()

def mean(xs):
    return sum(xs) / len(xs)


print("=== FOR __all__ ===")
print()
print("avg loss (r_SIMPLE) = \t{:.2%}".format(mean(data1)))
print("avg loss (r_COMPLEX) = \t{:.2%}".format(mean(data2)))
print()
print("avg loss delta = {:.2%}".format(mean(data1) - mean(data2)))
print(); print();

print("=== FOR __gt__(matching_normals) ===")
print()
print("avg loss (r_SIMPLE) = \t{:.2%}".format(mean(data1[3:])))
print("avg loss (r_COMPLEX) = \t{:.2%}".format(mean(data2[3:])))
print()
print("avg loss delta = {:.2%}".format(mean(data1[3:]) - mean(data2[3:])))
print(); print();
