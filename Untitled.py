import cairo
import math
import os
import imageio
import shutil
from scipy.integrate import quad
import sys
import numpy as np

iterations = 50

line_equation = lambda x, m, b : m*x + b

# functionscape = [[((x_path, (m, b)), (lower_bound,upper_bound))], [((y_path, (m, b)), (lower_bound,upper_bound))]]

path = [[0.7,0.9],[0.2,0.9],[0.2,0.5],[0.7,0.5],[0.7,0.1],[0.2,0.1],[0.701,0.901]]

total_path_length = 0
previous_point = []
for i in path:
    if path.index(i) == 0:
        previous_point = i
        continue

    x_diff = i[0] - previous_point[0]
    y_diff = i[1] - previous_point[1]


    path_length = math.sqrt((x_diff)**2 + (y_diff)**2)
    total_path_length += path_length

    previous_point = i

print(total_path_length)


functionscape = [[],[]]
previous_point = path[0]
lower_bound = 0
for i in path:
    if path.index(i) == 0:
        continue

    distance_from_last_point = math.sqrt((i[0] - previous_point[0])**2 + (i[1] - previous_point[1])**2)/total_path_length
    upper_bound = lower_bound + distance_from_last_point

    m_x = (i[0] - previous_point[0])/(upper_bound - lower_bound)
    b_x = i[0] - m_x*upper_bound
    functionscape[0].append(((line_equation, (m_x, b_x)), (lower_bound, upper_bound)))

    m_y = (i[1] - previous_point[1])/(upper_bound - lower_bound)
    b_y = i[1] - m_y*upper_bound
    functionscape[1].append(((line_equation, (m_y, b_y)), (lower_bound, upper_bound)))

    lower_bound = upper_bound
    previous_point = i

print('x: ')
for i in functionscape[0]:
    print(i[0][1])
    print(i[1])
    print(' ~ ')

print('y: ')
for i in functionscape[0]:
    print(i[0][1])
    print(i[1])
    print(' ~ ')

series = 'cos'

plotrange = 1

def f_by_cos(x,n,L,f,m,b):
    return f(x,m,b)*math.cos(n*math.pi*x/L)

def f_by_sin(x,n,L,f,m,b):
    return f(x,m,b)*math.sin(n*math.pi*x/L)

def half_range_sin_coefficient(L, n, functionscape):
    result = 0
    for i in functionscape:
        m = i[0][1][0]
        b = i[0][1][1]
        anon_func = i[0][0]
        result += quad(f_by_sin, i[1][0], i[1][1], args=(n,L,anon_func,m,b))[0]
    return 2*result/L

def first_half_range_cos_coefficient(L, functionscape):
    result = 0
    for i in functionscape:
        m = i[0][1][0]
        b = i[0][1][1]
        anon_func = i[0][0]
        result += quad(anon_func, i[1][0], i[1][1], args=(m,b))[0]
    return result/L

def consequent_half_range_cos_coefficient(L, n, functionscape):
    result = 0
    for i in functionscape:
        m = i[0][1][0]
        b = i[0][1][1]
        anon_func = i[0][0]
        result += quad(f_by_cos, i[1][0], i[1][1], args=(n,L,anon_func,m,b))[0]
    return 2*result/L



if (series == 'cos'):
    circle_dimensions = [(first_half_range_cos_coefficient(plotrange, functionscape[0]), 0, math.pi/2), (first_half_range_cos_coefficient(plotrange, functionscape[1]), 0, 0)]
elif (series == 'sin'):
    circle_dimensions = []

for n in range(1,iterations):
    if (series == 'cos'):
        coefficient = consequent_half_range_cos_coefficient(plotrange,n,functionscape[0])
        if (coefficient > 0.0001):
            circle_dimensions.append((500*coefficient, n/2, math.pi/2))
            circle_dimensions.append((500*coefficient, -n/2, math.pi/2))
        elif (coefficient < -0.0001):
            circle_dimensions.append((-500*coefficient, n/2, 3*math.pi/2))
            circle_dimensions.append((-500*coefficient, -n/2, 3*math.pi/2))

        coefficient = consequent_half_range_cos_coefficient(plotrange,n,functionscape[1])
        if (coefficient > 0.0001):
            circle_dimensions.append((500*coefficient, n/2, 0))
            circle_dimensions.append((500*coefficient, -n/2, 0))
        elif (coefficient < -0.0001):
            circle_dimensions.append((-500*coefficient, n/2, math.pi))
            circle_dimensions.append((-500*coefficient, -n/2, math.pi))

    elif (series == 'sin'):
        coefficient = half_range_sin_coefficient(plotrange,n,functionscape[0])
        if (coefficient > 0.0001):
            circle_dimensions.append((500*coefficient, n/2, 0))
            circle_dimensions.append((500*coefficient, -n/2, 0))
        elif (coefficient < -0.0001):
            circle_dimensions.append((-500*coefficient, n/2, math.pi))
            circle_dimensions.append((-500*coefficient, -n/2, math.pi))

        coefficient = half_range_sin_coefficient(plotrange,n,functionscape[1])
        if (coefficient > 0.0001):
            circle_dimensions.append((500*coefficient, n/2, math.pi/2))
            circle_dimensions.append((500*coefficient, -n/2, math.pi/2))
        elif (coefficient < -0.0001):
            circle_dimensions.append((-500*coefficient, n/2, 3*math.pi/2))
            circle_dimensions.append((-500*coefficient, -n/2, 3*math.pi/2))


for i in circle_dimensions:
    print(i)



# raise SystemExit

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''


FRAMECOUNT = 200

centre = (500,1600)

frame_dimensions = (2100,2100)

'''
circle_dimension[0] = radius of circle
circle_dimension[1] = number of times circle rotates around the origin.
'''

if (os.path.isdir(os.getcwd() + '/workdir')):
    print('~removing previous workdir~')
    shutil.rmtree(os.getcwd() + '/workdir/')

#find the sum of all x values (indice = 0) or y values (indice = 1) excluding the last one
def pointsSum(points, indice):
    sum = 0

    for i in points[:-1]:
        sum += i[indice]

    return sum

def fill_past_points(ctx, points_collection):
    ctx.set_source_rgb(1,1,1)
    for i in points_collection:
        ctx.arc(i[0], i[1], 5, 0, math.pi*2)
        ctx.fill()


points_collection = []

filenames = []
cwd = os.getcwd()

os.mkdir(cwd + '/workdir')

t = 0
onedxplot = []
onedyplot = []

for x in range(FRAMECOUNT):
    #create image surface and context
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,frame_dimensions[0],frame_dimensions[1])
    ctx = cairo.Context (surface)

    #black background
    ctx.set_source_rgb(0,0,0)
    ctx.rectangle(0,0,frame_dimensions[0],frame_dimensions[1])
    ctx.fill()

    #create center point
    ctx.set_source_rgb(1,1,1)
    ctx.arc(centre[0],centre[1],10,0,2*math.pi)
    ctx.fill()

    points = [centre]



    for i in circle_dimensions:

#         find angle offset
        if (series == 'sin'):
            offset = math.pi
        elif (series == 'cos'):
            offset = 3*math.pi/2

        theta = offset + (x/FRAMECOUNT)*2*math.pi*i[1] + i[2]

        xphase = i[0]*math.cos(theta)
        yphase = i[0]*math.sin(theta)
        points.append((xphase, yphase))

        finalx = pointsSum(points, 0) + xphase
        finaly = pointsSum(points, 1) + yphase

        #colour in point
        ctx.set_source_rgb(1,1,1)
        ctx.arc(finalx,finaly,5,0,2*math.pi)
        ctx.fill()

        ctx.arc(pointsSum(points, 0), pointsSum(points, 1), i[0], 0, 2*math.pi)
        ctx.stroke()

        #if we are on the last circle, fill in the past points and add the current one to the collection
        if i == circle_dimensions[-1]:
            fill_past_points(ctx, points_collection)
            points_collection.append((finalx, finaly))

    #horizontal and vertical tracking
    onedyplot.append((1100 + t, points_collection[-1][1])) 
    ctx.move_to(points_collection[-1][0], points_collection[-1][1])
    ctx.line_to(1100 + t, points_collection[-1][1])
    ctx.stroke()

    onedxplot.append((points_collection[-1][0], 1100 - t))
    ctx.move_to(points_collection[-1][0], points_collection[-1][1])
    ctx.line_to(points_collection[-1][0], 1100 - t)
    ctx.stroke()

    if (len(onedyplot) > 1):
        for k in range(len(onedyplot) - 1):
            ctx.move_to(onedyplot[k][0], onedyplot[k][1])
            ctx.line_to(onedyplot[k+1][0], onedyplot[k+1][1])
            ctx.stroke()

            ctx.move_to(onedxplot[k][0], onedxplot[k][1])
            ctx.line_to(onedxplot[k+1][0], onedxplot[k+1][1])
            ctx.stroke()



    surface.write_to_png('workdir/Frame' + str(x) + '.png')

    filenames.append(cwd + '/workdir/Frame' + str(x) + '.png')

    if (x % 10 == 0):
        print('doing frame ' + str(x))


    t += 1000/FRAMECOUNT

#create gif
images = []
for filename in filenames:
    images.append(imageio.imread(filename))

imageio.mimsave(cwd + '/final.gif', images)

#remove images after gif is created
shutil.rmtree(cwd + '/workdir/')
        
print('finished')
